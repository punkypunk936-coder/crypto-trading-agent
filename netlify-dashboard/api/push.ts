import { Buffer } from "node:buffer";
import { gunzipSync } from "node:zlib";

import {
  compareSnapshots,
  forwardNetlifyPush,
  json,
  persistCanonicalSnapshot,
  readGitCanonicalSnapshot,
  snapshotCycle,
  snapshotFromPayload,
  snapshotVersion,
  unauthorized,
} from "../lib/dashboard-store";

function decodePayload(data: any) {
  if (data?.encoding !== "gzip-base64") {
    return data;
  }
  if (typeof data.payload !== "string" || !data.payload) {
    throw new Error("Missing compressed payload");
  }
  const text = gunzipSync(Buffer.from(data.payload, "base64")).toString("utf8");
  return JSON.parse(text);
}

async function confirmGitCanonical(incoming: any) {
  const gitSnapshot = await readGitCanonicalSnapshot();
  if (!gitSnapshot?.state || compareSnapshots(gitSnapshot, incoming) < 0) {
    return null;
  }
  return gitSnapshot;
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }

  let data: any;
  try {
    data = decodePayload(await request.json());
  } catch (error) {
    return json(
      { ok: false, error: error instanceof Error ? error.message : "Invalid payload" },
      { status: 400 },
    );
  }
  if (!data || (!data.snapshot && !data.state)) {
    return json({ ok: false, error: "Missing snapshot/state in payload" }, { status: 400 });
  }

  const incoming = snapshotFromPayload(data);
  try {
    const { snapshot, componentErrors } = await persistCanonicalSnapshot(data);
    return json({
      ok: true,
      durable: true,
      storage: "vercel_blob",
      cycle: snapshotCycle(snapshot),
      version: snapshotVersion(snapshot),
      updatedAt: snapshot.updatedAt,
      component_warnings: componentErrors,
    });
  } catch (error) {
    const gitSnapshot = await confirmGitCanonical(incoming);
    if (gitSnapshot) {
      return json({
        ok: true,
        durable: true,
        storage: "github_canonical",
        cycle: snapshotCycle(gitSnapshot),
        version: snapshotVersion(gitSnapshot),
        updatedAt: gitSnapshot.updatedAt || null,
        blob_warning: error instanceof Error ? error.message : "Vercel Blob unavailable",
      });
    }

    const forwarded = await forwardNetlifyPush(data, request.headers.get("X-Token") || "");
    if (forwarded.ok) {
      return json(
        typeof forwarded.data === "object" && forwarded.data !== null
          ? { ...forwarded.data, fallback: "netlify", storage: "netlify_canonical" }
          : { ok: true, fallback: "netlify", storage: "netlify_canonical" },
      );
    }
    return json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Push failed",
        fallback_error: forwarded.data,
      },
      { status: 500 },
    );
  }
}

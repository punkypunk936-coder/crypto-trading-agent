import {
  PUSH_RECEIPT_PATH,
  cleanupChunkBlobs,
  compareSnapshots,
  deleteJson,
  json,
  persistCanonicalSnapshot,
  readGitCanonicalSnapshot,
  readJson,
  snapshotFromPayload,
  snapshotCycle,
  snapshotVersion,
  unauthorized,
  writeJson,
} from "../lib/dashboard-store";

function cleanSessionId(value: unknown) {
  return String(value || "").trim().replace(/[^a-zA-Z0-9._-]/g, "").slice(0, 96);
}

function chunkPath(sessionId: string, index: number) {
  return `dashboard/push-chunks/${sessionId}/${index}.json`;
}

function chunkPaths(sessionId: string, count: number) {
  return Array.from({ length: count }, (_, index) => chunkPath(sessionId, index));
}

async function githubReceipt(sessionId: string, chunkCount: number, warning: unknown, incoming: any) {
  const gitSnapshot = await readGitCanonicalSnapshot();
  if (!gitSnapshot?.state || compareSnapshots(gitSnapshot, incoming) < 0) return null;
  return {
    ok: true,
    assembled: true,
    durable: true,
    storage: "github_canonical",
    session_id: sessionId,
    chunks: chunkCount,
    cycle: snapshotCycle(gitSnapshot),
    version: snapshotVersion(gitSnapshot),
    updatedAt: gitSnapshot.updatedAt || null,
    blob_warning: warning instanceof Error ? warning.message : String(warning || "Vercel Blob unavailable"),
  };
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }

  const data = await request.json().catch(() => null);
  const sessionId = cleanSessionId(data?.session_id);
  const chunkIndex = Number.parseInt(String(data?.chunk_index ?? ""), 10);
  const chunkCount = Number.parseInt(String(data?.chunk_count ?? ""), 10);
  const chunk = typeof data?.chunk === "string" ? data.chunk : "";
  const announcedVersion = Number(data?.version || 0);
  const announcedCycle = Number(data?.cycle_number || 0);
  const announcedSnapshot = announcedVersion > 0 || announcedCycle > 0
    ? {
        version: announcedVersion,
        updatedAt: String(data?.updatedAt || ""),
        state: { cycle_number: announcedCycle },
      }
    : null;

  if (!sessionId || !Number.isInteger(chunkIndex) || !Number.isInteger(chunkCount) || chunkCount <= 0 || chunkIndex < 0 || chunkIndex >= chunkCount || !chunk) {
    return json({ ok: false, error: "Invalid chunk payload" }, { status: 400 });
  }
  if (chunkCount > 128) {
    return json({ ok: false, error: "Too many chunks" }, { status: 400 });
  }

  try {
    const priorReceipt = await readJson(PUSH_RECEIPT_PATH, null);
    if (priorReceipt?.session_id === sessionId && priorReceipt?.assembled) {
      return json(priorReceipt);
    }
    if (chunkIndex === 0) {
      await cleanupChunkBlobs(sessionId);
    }
    await writeJson(chunkPath(sessionId, chunkIndex), { chunk });

    const pieces: string[] = [];
    const missing: number[] = [];
    for (let index = 0; index < chunkCount; index += 1) {
      if (index === chunkIndex) {
        pieces.push(chunk);
        continue;
      }
      const piece = await readJson(chunkPath(sessionId, index), null);
      if (!piece || typeof piece.chunk !== "string") {
        missing.push(index);
      } else {
        pieces.push(piece.chunk);
      }
    }

    if (missing.length) {
      return json({
        ok: true,
        assembled: false,
        session_id: sessionId,
        received_index: chunkIndex,
        missing,
      }, { status: 202 });
    }

    const payloadText = pieces.join("");
    const payload = JSON.parse(payloadText);
    const incoming = snapshotFromPayload(payload);
    const { snapshot, componentErrors } = await persistCanonicalSnapshot(payload);
    const receipt = {
      ok: true,
      assembled: true,
      durable: true,
      storage: "vercel_blob",
      session_id: sessionId,
      chunks: chunkCount,
      bytes: payloadText.length,
      cycle: snapshotCycle(snapshot),
      version: snapshotVersion(snapshot),
      updatedAt: snapshot.updatedAt,
      component_warnings: componentErrors,
    };
    await writeJson(PUSH_RECEIPT_PATH, receipt);
    await deleteJson(chunkPaths(sessionId, chunkCount));
    return json(receipt);
  } catch (error) {
    if (chunkIndex === chunkCount - 1) {
      let incoming = null;
      try {
        const pieces = await Promise.all(
          chunkPaths(sessionId, chunkCount).map(async (pathname, index) => {
            if (index === chunkIndex) return chunk;
            return String((await readJson(pathname, null))?.chunk || "");
          }),
        );
        if (pieces.every(Boolean)) {
          incoming = snapshotFromPayload(JSON.parse(pieces.join("")));
        }
      } catch {
        incoming = null;
      }
      const receipt = incoming || announcedSnapshot
        ? await githubReceipt(sessionId, chunkCount, error, incoming || announcedSnapshot)
        : null;
      if (receipt) return json(receipt);
    }
    return json({
      ok: true,
      assembled: false,
      durable: true,
      storage: "github_canonical",
      session_id: sessionId,
      received_index: chunkIndex,
      waiting_for_final_chunk: true,
      blob_warning: error instanceof Error ? error.message : "Vercel Blob unavailable",
    }, { status: 202 });
  }
}

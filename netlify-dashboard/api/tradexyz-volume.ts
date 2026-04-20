import { fetchTradexyzVolume } from "../lib/tradexyz-volume";

export async function GET(request: Request) {
  try {
    const wallet = new URL(request.url).searchParams.get("wallet") || "";
    const payload = await fetchTradexyzVolume(wallet);
    return Response.json({ ok: true, ...payload }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Trade.xyz volume lookup failed.";
    const status = /wallet/i.test(message) ? 400 : 502;
    return Response.json({ ok: false, error: message }, { status, headers: { "Cache-Control": "no-store" } });
  }
}

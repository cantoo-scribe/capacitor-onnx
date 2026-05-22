import { FileTransfer } from "@capacitor/file-transfer";
import { Directory, Filesystem } from "@capacitor/filesystem";

export type LocalModelStatus = "cache_hit" | "downloaded";

export interface LocalModel {
  /** Absolute `file://` URI of the model on the device filesystem. */
  path: string;
  status: LocalModelStatus;
}

/** App-private, persistent directory — not evicted by the OS, not user-visible. */
const MODEL_DIRECTORY = Directory.Data;

/** Maps a model URL to a deterministic relative path under the app data directory. */
function relativePathForUrl(url: string): string {
  const cleaned = new URL(url).pathname.replace(/^\/+/, "");
  if (!cleaned) {
    throw new Error(`Cannot derive a local model path from URL: ${url}`);
  }
  return cleaned;
}

function parentDirectory(path: string): string {
  const idx = path.lastIndexOf("/");
  return idx > 0 ? path.slice(0, idx) : "";
}

async function modelFileExists(relativePath: string): Promise<boolean> {
  try {
    const stat = await Filesystem.stat({
      directory: MODEL_DIRECTORY,
      path: relativePath,
    });
    return stat.type === "file" && stat.size > 0;
  } catch {
    // stat throws when the file does not exist.
    return false;
  }
}

/**
 * Ensures the model at `url` exists on the native filesystem and returns its
 * absolute path. An already-downloaded file is reused instead of re-downloaded.
 *
 * Native only — the web flow fetches the model into a `modelBuffer` instead.
 */
export async function ensureLocalModel(url: string): Promise<LocalModel> {
  const relativePath = relativePathForUrl(url);
  const { uri } = await Filesystem.getUri({
    directory: MODEL_DIRECTORY,
    path: relativePath,
  });

  if (await modelFileExists(relativePath)) {
    return { path: uri, status: "cache_hit" };
  }

  const dir = parentDirectory(relativePath);
  if (dir) {
    try {
      await Filesystem.mkdir({
        directory: MODEL_DIRECTORY,
        path: dir,
        recursive: true,
      });
    } catch {
      // Directory already exists — ignore; a real failure surfaces on download.
    }
  }

  await FileTransfer.downloadFile({ url, path: uri });

  return { path: uri, status: "downloaded" };
}

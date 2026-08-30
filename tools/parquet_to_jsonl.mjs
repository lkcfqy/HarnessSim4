import { createWriteStream } from "node:fs";
import { readdir } from "node:fs/promises";
import { once } from "node:events";
import path from "node:path";
import process from "node:process";
import { asyncBufferFromFile, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";

const [datasetDir, outputPath] = process.argv.slice(2);
if (!datasetDir || !outputPath) {
  throw new Error("usage: node tools/parquet_to_jsonl.mjs DATASET_DIR OUTPUT.jsonl");
}

const chunkRoot = path.join(datasetDir, "data");
const chunks = (await readdir(chunkRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory() && entry.name.startsWith("chunk-"))
  .map((entry) => entry.name)
  .sort();

const files = [];
for (const chunk of chunks) {
  const chunkDir = path.join(chunkRoot, chunk);
  const names = (await readdir(chunkDir))
    .filter((name) => name.endsWith(".parquet"))
    .sort();
  files.push(...names.map((name) => path.join(chunkDir, name)));
}
if (files.length === 0) {
  throw new Error(`No parquet files found below ${chunkRoot}`);
}

const stream = createWriteStream(outputPath, { encoding: "utf8" });
const replacer = (_key, value) => (typeof value === "bigint" ? Number(value) : value);
let rowCount = 0;
for (const filePath of files) {
  const file = await asyncBufferFromFile(filePath);
  const rows = await parquetReadObjects({
    file,
    compressors,
    columns: ["observation.state", "action", "episode_index", "frame_index", "timestamp"],
  });
  for (const row of rows) {
    const normalized = {
      "observation.state": row["observation.state"] ?? row.observation?.state,
      action: row.action,
      episode_index: row.episode_index,
      frame_index: row.frame_index,
      timestamp: row.timestamp,
    };
    if (!Array.isArray(normalized["observation.state"]) || !Array.isArray(normalized.action)) {
      throw new Error(`Nested state/action columns were not reconstructed in ${filePath}`);
    }
    if (!stream.write(`${JSON.stringify(normalized, replacer)}\n`)) {
      await once(stream, "drain");
    }
    rowCount += 1;
  }
}
stream.end();
await once(stream, "finish");
process.stdout.write(JSON.stringify({ files: files.length, rows: rowCount, outputPath }));

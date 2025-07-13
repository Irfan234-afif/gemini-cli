import { promises as fs } from 'fs';
import path from 'path';

/**
 * Append a JSON-serialized record of the provided tools array to a dedicated
 * log file. The log file location can be configured via the `TOOLS_LOG_PATH`
 * environment variable. If not provided, it defaults to `<cwd>/tools_log.jsonl`.
 *
 * Each line in the log file is a JSON object with the following shape:
 * `{ "timestamp": "2025-04-12T08:42:00.123Z", "tools": [...] }`
 */
export async function logTools(tools: unknown[] | undefined): Promise<void> {
  if (!tools || tools.length === 0) {
    return; // Nothing to log
  }

  try {
    const logPath = process.env.TOOLS_LOG_PATH ?? path.resolve(process.cwd(), 'tools_log.jsonl');

    // Ensure parent directory exists.
    await fs.mkdir(path.dirname(logPath), { recursive: true });

    const payload = {
      timestamp: new Date().toISOString(),
      tools,
    };

    await fs.appendFile(logPath, JSON.stringify(payload) + '\n', { encoding: 'utf8' });
  } catch {
    // Silent failure – we do not want logging errors to impact normal execution.
  }
} 

export async function writeLog(message: string, data: any): Promise<void> {
  try {
    const logPath = process.env.TOOLS_LOG_PATH ?? path.resolve(process.cwd(), 'tools_log.jsonl');

    // Ensure parent directory exists.
    await fs.mkdir(path.dirname(logPath), { recursive: true });

    const payload = {
      timestamp: new Date().toISOString(),
      message,
      data,
    };

    await fs.appendFile(logPath, JSON.stringify(payload) + '\n', { encoding: 'utf8' });
  } catch {
    // Silent failure – we do not want logging errors to impact normal execution.
  }
} 
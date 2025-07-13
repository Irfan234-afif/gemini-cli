import fs from 'fs';
import { promises as fsPromises } from 'fs';
import path from 'path';

// Ensure the `logs` directory exists at startup.
const logsDir = path.resolve(process.cwd(), 'logs');
try {
  fs.mkdirSync(logsDir, { recursive: true });
} catch {
  // Directory already exists or cannot be created – ignore.
}

/**
 * Construct a log-file name for the current day.
 * The format is `app-YYYY-MM-DD.log`.
 */
const getLogFileName = (): string => {
  const now = new Date();
  const date = now.toISOString().split('T')[0];
  return `app-${date}.log`;
};

/**
 * Return the current timestamp as an ISO string.
 */
const getTimestamp = (): string => new Date().toISOString();

/**
 * Append a log entry to a file. If writing fails it is silently ignored so as
 * not to interrupt normal program execution.
 */
const writeToFile = async (
  level: 'info' | 'warn' | 'error' | 'debug',
  message: string,
  data: unknown = undefined,
  logFileName: string = getLogFileName(),
): Promise<void> => {
  const timestamp = getTimestamp();
  const logFilePath = path.join(logsDir, logFileName);

  let logEntry = `[${timestamp}] [${level.toUpperCase()}] ${message}`;

  if (data !== undefined && data !== null) {
    const serialised = typeof data === 'object' ? JSON.stringify(data, null, 2) : String(data);
    logEntry += `\nData: ${serialised}`;
  }

  logEntry += '\n\n';

  try {
    await fsPromises.appendFile(logFilePath, logEntry, { encoding: 'utf8' });
  } catch {
    // If we cannot write to the file, fall back to console so that the problem
    // is at least visible during development.
    console.error('Failed to write to log file');
  }

  // Mirror log entry to the console when in development mode.
  if (process.env.NODE_ENV === 'development') {
    const consoleMethod = (console as any)[level] ?? console.log;
    consoleMethod(`[${level.toUpperCase()}] ${message}`, data);
  }
};

export const logger = {
  info: (message: string, data?: unknown, logFileName?: string) =>
    writeToFile('info', message, data, logFileName),
  warn: (message: string, data?: unknown, logFileName?: string) =>
    writeToFile('warn', message, data, logFileName),
  error: (message: string, data?: unknown, logFileName?: string) =>
    writeToFile('error', message, data, logFileName),
  debug: (message: string, data?: unknown, logFileName?: string) => {
    if (process.env.NODE_ENV === 'development') {
      return writeToFile('debug', message, data, logFileName);
    }
  },
};

/**
 * Log an incoming HTTP request (and optionally its corresponding response).
 */
export const logRequest = (
  request: Request,
  response: Response | null = null,
): void => {
  const url = new URL(request.url);
  const logData: Record<string, unknown> = {
    method: request.method,
    url: url.pathname,
    query: Object.fromEntries(url.searchParams),
    headers: Object.fromEntries(request.headers.entries()),
    timestamp: getTimestamp(),
  };

  if (response) {
    logData.responseStatus = response.status;
    logData.responseHeaders = Object.fromEntries(response.headers.entries());
  }

  logger.info('HTTP Request', logData);
};

/** Log database operations for auditing purposes. */
export const logDatabase = (
  operation: string,
  table: string,
  data: unknown = undefined,
): void => {
  logger.info('Database Operation', {
    operation,
    table,
    data,
    timestamp: getTimestamp(),
  });
};

/** Log Shopify API calls for troubleshooting or analytics. */
export const logShopifyAPI = (
  operation: string,
  data: unknown = undefined,
): void => {
  logger.info('Shopify API Call', {
    operation,
    data,
    timestamp: getTimestamp(),
  });
};

export default logger;
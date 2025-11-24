import type { DatabaseInfo, Program, MetaFile, MetaContent } from '../types';

const API_BASE = '';

export async function listDatabases(): Promise<DatabaseInfo[]> {
  const response = await fetch(`${API_BASE}/list_databases`);
  if (!response.ok) {
    throw new Error(`Failed to load database list (HTTP ${response.status})`);
  }
  return response.json();
}

export async function getPrograms(dbPath: string): Promise<Program[]> {
  const response = await fetch(
    `${API_BASE}/get_programs?db_path=${encodeURIComponent(dbPath)}`
  );
  if (!response.ok) {
    if (response.status === 503) {
      throw new Error(
        'Database temporarily unavailable - evolution may be running'
      );
    }
    throw new Error(`Failed to load data (HTTP ${response.status})`);
  }
  return response.json();
}

export async function getMetaFiles(dbPath: string): Promise<MetaFile[]> {
  const response = await fetch(
    `${API_BASE}/get_meta_files?db_path=${encodeURIComponent(dbPath)}`
  );
  if (!response.ok) {
    if (response.status === 404) {
      return [];
    }
    throw new Error(`Failed to load meta files (HTTP ${response.status})`);
  }
  return response.json();
}

export async function getMetaContent(
  dbPath: string,
  generation: number
): Promise<MetaContent> {
  const response = await fetch(
    `${API_BASE}/get_meta_content?db_path=${encodeURIComponent(dbPath)}&generation=${generation}`
  );
  if (!response.ok) {
    throw new Error(`Failed to load meta content (HTTP ${response.status})`);
  }
  return response.json();
}

export function getMetaPdfUrl(dbPath: string, generation: number): string {
  return `${API_BASE}/download_meta_pdf?db_path=${encodeURIComponent(dbPath)}&generation=${generation}`;
}

/**
 * API Service — バックエンドとの通信を管理.
 */

const API_BASE = '/api';

export interface UploadedFile {
  id: string;
  original_name: string;
  stored_path: string;
  file_size: number;
  mime_type: string | null;
}

export interface EngineCapabilities {
  supports_single_image: boolean;
  supports_multi_image: boolean;
  supports_cad_input: boolean;
  outputs_mesh: boolean;
  outputs_cad: boolean;
  outputs_point_cloud: boolean;
  supported_output_formats: string[];
  requires_gpu: boolean;
  estimated_vram_gb: number | null;
}

export interface WeightFileInfo {
  name: string;
  url: string;
  relative_path: string;
  description: string;
  requires_auth: boolean;
}

export interface EngineInfo {
  name: string;
  display_name: string;
  description: string;
  version: string;
  capabilities: EngineCapabilities;
  status: 'ready' | 'weights_missing' | 'dependency_missing' | 'error';
  required_weights: WeightFileInfo[];
  readme_path: string | null;
}

export interface GenerationProgress {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  error: string | null;
}

export interface APIResponse<T = unknown> {
  success: boolean;
  message: string;
  data: T;
}

// ── Upload ──────────────────────

export async function uploadFiles(files: File[]): Promise<UploadedFile[]> {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  const res = await fetch(`${API_BASE}/upload/`, { method: 'POST', body: formData });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Upload failed');
  }
  const json: APIResponse<UploadedFile[]> = await res.json();
  return json.data;
}

export async function deleteFile(fileId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/upload/${fileId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Delete failed');
}

// ── Engines ─────────────────────

export async function listEngines(): Promise<EngineInfo[]> {
  const res = await fetch(`${API_BASE}/generate/engines`);
  if (!res.ok) throw new Error('Failed to list engines');
  return res.json();
}

export async function getEngineReadme(engineName: string): Promise<string> {
  const res = await fetch(`${API_BASE}/generate/engines/${engineName}/readme`);
  if (!res.ok) throw new Error('Failed to get README');
  const json: APIResponse<{ content: string }> = await res.json();
  return json.data.content;
}

// ── Weights ─────────────────────

export async function getWeightStatus(engineName: string): Promise<unknown[]> {
  const res = await fetch(`${API_BASE}/weights/${engineName}`);
  if (!res.ok) throw new Error('Failed to get weight status');
  const json: APIResponse<unknown[]> = await res.json();
  return json.data;
}

export async function downloadWeights(engineName: string): Promise<APIResponse> {
  const res = await fetch(`${API_BASE}/weights/${engineName}/download`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to download weights');
  return res.json();
}

// ── Generation ──────────────────

export interface GenerationRequest {
  engine_name: string;
  images: Array<{
    file_id: string;
    view_angle: string;
    custom_azimuth?: number;
    custom_elevation?: number;
  }>;
  output_format: string;
  engine_params: Record<string, unknown>;
}

export async function startGeneration(request: GenerationRequest): Promise<string> {
  const res = await fetch(`${API_BASE}/generate/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to start generation');
  }
  const json: APIResponse<{ job_id: string }> = await res.json();
  return json.data.job_id;
}

export async function getJobStatus(jobId: string): Promise<GenerationProgress> {
  const res = await fetch(`${API_BASE}/generate/jobs/${jobId}`);
  if (!res.ok) throw new Error('Failed to get job status');
  return res.json();
}

// ── Export ───────────────────────

export function getDownloadUrl(jobId: string, format?: string): string {
  let url = `${API_BASE}/export/download/${jobId}`;
  if (format) url += `?format=${format}`;
  return url;
}

export async function openExternal(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/export/open-external/${jobId}`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to open in external app');
}

export async function reloadFromExternal(jobId: string): Promise<unknown> {
  const res = await fetch(`${API_BASE}/export/reload/${jobId}`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to reload');
  const json: APIResponse = await res.json();
  return json.data;
}

// ── Settings ────────────────────

export async function getSettings(): Promise<unknown> {
  const res = await fetch(`${API_BASE}/settings/`);
  if (!res.ok) throw new Error('Failed to get settings');
  const json: APIResponse = await res.json();
  return json.data;
}

export async function updateProxy(data: { http_proxy: string | null; https_proxy: string | null }): Promise<void> {
  const res = await fetch(`${API_BASE}/settings/proxy`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update proxy');
}

export async function updateHuggingFace(data: { token: string | null; cache_dir: string | null }): Promise<void> {
  const res = await fetch(`${API_BASE}/settings/huggingface`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update HF settings');
}

// ── WebSocket ───────────────────

export function connectProgressWS(
  jobId: string,
  onMessage: (progress: GenerationProgress) => void,
  onClose?: () => void,
): WebSocket {
  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${wsProto}//${window.location.host}${API_BASE}/generate/ws/${jobId}`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  ws.onclose = () => onClose?.();
  return ws;
}

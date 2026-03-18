/**
 * CAD3D Generator — メインアプリコンポーネント.
 * 
 * 2D画像/CADから3D CADモデルを生成するWebアプリのUI.
 * グラスモーフィズムのダークテーマ。
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import * as api from './services/api'

// ── 型定義 ─────────────────────────────────────────

interface LocalFile {
  id: string;
  name: string;
  size: number;
  viewAngle: string;
}

type TabId = 'input' | 'engine' | 'weights' | 'settings';

// ── App ────────────────────────────────────────────

export default function App() {
  // State
  const [files, setFiles] = useState<LocalFile[]>([]);
  const [engines, setEngines] = useState<api.EngineInfo[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<string>('');
  const [outputFormat, setOutputFormat] = useState('glb');
  const [activeTab, setActiveTab] = useState<TabId>('input');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState<api.GenerationProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [readmeContent, setReadmeContent] = useState<string>('');
  const [showReadme, setShowReadme] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [toast, setToast] = useState<{ type: string; msg: string } | null>(null);

  // Settings state
  const [proxyHttp, setProxyHttp] = useState('');
  const [proxyHttps, setProxyHttps] = useState('');
  const [hfToken, setHfToken] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Effects ──────────────────────────────

  useEffect(() => {
    loadEngines();
  }, []);

  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(t);
    }
  }, [toast]);

  // ── Handlers ─────────────────────────────

  const loadEngines = async () => {
    try {
      const list = await api.listEngines();
      setEngines(list);
      if (list.length > 0 && !selectedEngine) {
        setSelectedEngine(list[0].name);
      }
    } catch {
      // エンジン未発見（初回起動時は正常）
    }
  };

  const showToast = (type: string, msg: string) => setToast({ type, msg });

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (!selected || selected.length === 0) return;
    await uploadFiles(Array.from(selected));
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const items = Array.from(e.dataTransfer.files);
    if (items.length > 0) await uploadFiles(items);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const uploadFiles = async (rawFiles: File[]) => {
    try {
      const uploaded = await api.uploadFiles(rawFiles);
      const newFiles: LocalFile[] = uploaded.map(u => ({
        id: u.id,
        name: u.original_name,
        size: u.file_size,
        viewAngle: 'unknown',
      }));
      setFiles(prev => [...prev, ...newFiles]);
      showToast('success', `${uploaded.length} ファイルをアップロードしました`);
    } catch (e: unknown) {
      showToast('error', `アップロード失敗: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
  };

  const removeFile = async (id: string) => {
    try {
      await api.deleteFile(id);
      setFiles(prev => prev.filter(f => f.id !== id));
    } catch {
      showToast('error', 'ファイル削除に失敗しました');
    }
  };

  const setViewAngle = (id: string, angle: string) => {
    setFiles(prev => prev.map(f => f.id === id ? { ...f, viewAngle: angle } : f));
  };

  const startGeneration = async () => {
    if (!selectedEngine || files.length === 0) {
      showToast('warning', 'エンジンと画像を選択してください');
      return;
    }

    setIsGenerating(true);
    setProgress(null);

    try {
      const jid = await api.startGeneration({
        engine_name: selectedEngine,
        images: files.map(f => ({
          file_id: f.id,
          view_angle: f.viewAngle,
        })),
        output_format: outputFormat,
        engine_params: {},
      });
      setJobId(jid);

      // WebSocket で進捗監視
      const ws = api.connectProgressWS(jid, (p) => {
        setProgress(p);
        if (p.status === 'completed') {
          setIsGenerating(false);
          showToast('success', '3Dモデルの生成が完了しました！');
        } else if (p.status === 'failed') {
          setIsGenerating(false);
          showToast('error', p.error || '生成に失敗しました');
        }
      }, () => {
        // WS接続失敗時のポーリングフォールバック
        if (isGenerating) pollJob(jid);
      });

      return () => ws.close();
    } catch (e: unknown) {
      setIsGenerating(false);
      showToast('error', `生成開始失敗: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
  };

  const pollJob = async (jid: string) => {
    const poll = async () => {
      try {
        const p = await api.getJobStatus(jid);
        setProgress(p);
        if (p.status === 'completed') {
          setIsGenerating(false);
          showToast('success', '3Dモデルの生成が完了しました！');
        } else if (p.status === 'failed') {
          setIsGenerating(false);
          showToast('error', p.error || '生成に失敗しました');
        } else {
          setTimeout(poll, 1000);
        }
      } catch {
        setTimeout(poll, 2000);
      }
    };
    poll();
  };

  const openReadme = async (engineName: string) => {
    try {
      const content = await api.getEngineReadme(engineName);
      setReadmeContent(content);
      setShowReadme(true);
    } catch {
      showToast('error', 'README の取得に失敗しました');
    }
  };

  const handleDownloadWeights = async (engineName: string) => {
    showToast('info', 'ダウンロード中...');
    try {
      await api.downloadWeights(engineName);
      showToast('success', 'ダウンロード完了');
      loadEngines();
    } catch (e: unknown) {
      showToast('error', `ダウンロード失敗: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
  };

  const handleSaveSettings = async () => {
    try {
      await api.updateProxy({ http_proxy: proxyHttp || null, https_proxy: proxyHttps || null });
      if (hfToken) await api.updateHuggingFace({ token: hfToken, cache_dir: null });
      showToast('success', '設定を保存しました');
      setShowSettings(false);
    } catch {
      showToast('error', '設定の保存に失敗しました');
    }
  };

  const downloadResult = () => {
    if (jobId) window.open(api.getDownloadUrl(jobId, outputFormat), '_blank');
  };

  const openInExternal = async () => {
    if (!jobId) return;
    try {
      await api.openExternal(jobId);
      showToast('success', '外部アプリで開きました');
    } catch {
      showToast('error', '外部アプリでの起動に失敗しました');
    }
  };

  const reloadModel = async () => {
    if (!jobId) return;
    try {
      await api.reloadFromExternal(jobId);
      showToast('success', 'モデルを再読み込みしました');
    } catch {
      showToast('error', '再読み込みに失敗しました');
    }
  };

  // ── Helper ───────────────────────────────

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const selectedEngineInfo = engines.find(e => e.name === selectedEngine);
  const isCompleted = progress?.status === 'completed';

  // ── Render ───────────────────────────────

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="app-logo">
          <span style={{ fontSize: '1.5rem' }}>◆</span>
          <h1>CAD3D Generator</h1>
          <span className="version">v0.1.0</span>
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary btn-icon" onClick={() => setShowSettings(true)} title="設定">
            ⚙
          </button>
        </div>
      </header>

      <div className="app-main">
        {/* Side Panel */}
        <div className="side-panel">
          {/* Tabs */}
          <div style={{ padding: 'var(--space-sm)' }}>
            <div className="tabs">
              <button className={`tab ${activeTab === 'input' ? 'active' : ''}`} onClick={() => setActiveTab('input')}>
                📁 入力
              </button>
              <button className={`tab ${activeTab === 'engine' ? 'active' : ''}`} onClick={() => setActiveTab('engine')}>
                🔧 エンジン
              </button>
              <button className={`tab ${activeTab === 'weights' ? 'active' : ''}`} onClick={() => setActiveTab('weights')}>
                📦 重み
              </button>
            </div>
          </div>

          {/* Tab Content: Input */}
          {activeTab === 'input' && (
            <>
              <div className="panel-section">
                <div className="panel-section-title">📤 ファイル入力</div>
                <div
                  className="dropzone"
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                >
                  <div className="dropzone-icon">📷</div>
                  <div className="dropzone-text">
                    <strong>クリック</strong>または<strong>ドラッグ＆ドロップ</strong><br />
                    で画像/CADファイルを追加<br />
                    <small>JPG, PNG, PDF, DXF, SVG, TIFF, HEIC 等</small>
                  </div>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.pdf,.dxf,.svg,.heic,.webp"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />

                {files.length > 0 && (
                  <div className="file-list">
                    {files.map(f => (
                      <div key={f.id} className="file-item">
                        <span className="file-item-icon">📄</span>
                        <span className="file-item-name">{f.name}</span>
                        <span className="file-item-size">{formatSize(f.size)}</span>
                        <select
                          value={f.viewAngle}
                          onChange={(e) => setViewAngle(f.id, e.target.value)}
                          className="form-select"
                          style={{ width: '80px', padding: '2px 4px', fontSize: '0.7rem' }}
                          title="視点角度を指定"
                        >
                          <option value="unknown">自動</option>
                          <option value="front">正面</option>
                          <option value="back">背面</option>
                          <option value="left">左側</option>
                          <option value="right">右側</option>
                          <option value="top">上面</option>
                          <option value="bottom">底面</option>
                          <option value="isometric">等角</option>
                        </select>
                        <button className="file-item-delete" onClick={() => removeFile(f.id)} title="削除">
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Output Format */}
              <div className="panel-section">
                <div className="panel-section-title">📐 出力設定</div>
                <div className="form-group">
                  <label className="form-label">出力フォーマット</label>
                  <select className="form-select" value={outputFormat} onChange={e => setOutputFormat(e.target.value)}>
                    <option value="glb">GLB (Web3D)</option>
                    <option value="obj">OBJ (メッシュ)</option>
                    <option value="stl">STL (3Dプリント)</option>
                    <option value="ply">PLY (点群)</option>
                    <option value="gltf">glTF (JSON)</option>
                    <option value="step">STEP (CAD)</option>
                  </select>
                </div>
              </div>

              {/* Generate Button */}
              <div className="panel-section" style={{ borderBottom: 'none' }}>
                <button
                  className="btn btn-primary btn-full btn-generate"
                  disabled={files.length === 0 || !selectedEngine || isGenerating}
                  onClick={startGeneration}
                >
                  {isGenerating ? (
                    <><span className="animate-spin">⟳</span> 生成中...</>
                  ) : (
                    <>🚀 3Dモデルを生成</>
                  )}
                </button>
              </div>
            </>
          )}

          {/* Tab Content: Engine */}
          {activeTab === 'engine' && (
            <div className="panel-section">
              <div className="panel-section-title">🔧 生成エンジン選択</div>
              {engines.length === 0 ? (
                <p style={{ color: 'var(--color-text-tertiary)', fontSize: '0.85rem' }}>
                  エンジンが見つかりません。バックエンドが起動しているか確認してください。
                </p>
              ) : (
                engines.map(engine => (
                  <div
                    key={engine.name}
                    className={`engine-card ${selectedEngine === engine.name ? 'selected' : ''}`}
                    onClick={() => setSelectedEngine(engine.name)}
                  >
                    <div className="engine-card-header">
                      <span className="engine-card-name">{engine.display_name}</span>
                      <span className={`engine-status ${engine.status === 'ready' ? 'ready' : engine.status === 'weights_missing' ? 'missing' : 'error'}`}>
                        {engine.status === 'ready' ? '✓ Ready' : engine.status === 'weights_missing' ? '⚠ 重み不足' : '✕ Error'}
                      </span>
                    </div>
                    <p className="engine-card-desc">{engine.description}</p>
                    <div className="engine-card-caps">
                      {engine.capabilities.supports_single_image && <span className="cap-badge">1画像</span>}
                      {engine.capabilities.supports_multi_image && <span className="cap-badge">複数画像</span>}
                      {engine.capabilities.outputs_mesh && <span className="cap-badge">メッシュ</span>}
                      {engine.capabilities.outputs_cad && <span className="cap-badge">CAD</span>}
                      {engine.capabilities.requires_gpu && <span className="cap-badge">GPU</span>}
                      {engine.capabilities.estimated_vram_gb && (
                        <span className="cap-badge">VRAM {engine.capabilities.estimated_vram_gb}GB</span>
                      )}
                    </div>
                    <div style={{ marginTop: 'var(--space-sm)', display: 'flex', gap: 'var(--space-xs)' }}>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                        onClick={(e) => { e.stopPropagation(); openReadme(engine.name); }}
                      >
                        📖 マニュアル
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab Content: Weights */}
          {activeTab === 'weights' && (
            <div className="panel-section">
              <div className="panel-section-title">📦 モデル重み管理</div>
              {selectedEngineInfo ? (
                <>
                  <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-md)' }}>
                    <strong>{selectedEngineInfo.display_name}</strong> の重みファイル
                  </p>
                  {selectedEngineInfo.required_weights.map((w, i) => (
                    <div key={i} className="weight-item">
                      <div className="weight-status-dot missing" />
                      <span className="weight-name">{w.name}</span>
                      <button
                        className="btn btn-secondary"
                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}
                        onClick={() => handleDownloadWeights(selectedEngineInfo.name)}
                      >
                        ダウンロード
                      </button>
                    </div>
                  ))}
                  <button
                    className="btn btn-primary btn-full"
                    style={{ marginTop: 'var(--space-md)' }}
                    onClick={() => handleDownloadWeights(selectedEngineInfo.name)}
                  >
                    📥 全てダウンロード
                  </button>
                  <button
                    className="btn btn-secondary btn-full"
                    style={{ marginTop: 'var(--space-sm)' }}
                    onClick={() => openReadme(selectedEngineInfo.name)}
                  >
                    📖 手動配置ガイド
                  </button>
                </>
              ) : (
                <p style={{ color: 'var(--color-text-tertiary)', fontSize: '0.85rem' }}>
                  エンジンを選択してください。
                </p>
              )}
            </div>
          )}
        </div>

        {/* Viewer Panel */}
        <div className="viewer-panel">
          {/* Toolbar */}
          <div className="viewer-toolbar">
            <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
              3D プレビュー
            </span>
            <div style={{ flex: 1 }} />
            {isCompleted && (
              <>
                <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '4px 10px' }} onClick={downloadResult}>
                  💾 ダウンロード
                </button>
                <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '4px 10px' }} onClick={openInExternal}>
                  🔗 外部CADで開く
                </button>
                <button className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '4px 10px' }} onClick={reloadModel}>
                  🔄 再読み込み
                </button>
              </>
            )}
          </div>

          {/* 3D Viewer Area */}
          <div className="viewer-canvas">
            {isGenerating && progress && (
              <div className="progress-overlay">
                <div className="progress-percentage">
                  {Math.round(progress.progress * 100)}%
                </div>
                <div className="progress-bar-container">
                  <div className="progress-bar" style={{ width: `${progress.progress * 100}%` }} />
                </div>
                <div className="progress-message">{progress.message}</div>
              </div>
            )}

            {!isCompleted && !isGenerating && (
              <div className="viewer-empty">
                <div className="viewer-empty-icon">◇</div>
                <div className="viewer-empty-text">3Dビューア</div>
                <div className="viewer-empty-sub">
                  左パネルから画像をアップロードし、エンジンを選択して「3Dモデルを生成」を押してください。
                  生成されたモデルがここに表示されます。
                </div>
              </div>
            )}

            {isCompleted && (
              <div className="viewer-empty">
                <div className="viewer-empty-icon" style={{ color: 'var(--color-success)' }}>✓</div>
                <div className="viewer-empty-text" style={{ color: 'var(--color-success)' }}>
                  生成完了！
                </div>
                <div className="viewer-empty-sub">
                  上部のボタンからダウンロード、外部CADで開く、再読み込みが可能です。<br />
                  <small style={{ color: 'var(--color-text-tertiary)' }}>
                    ※ Three.jsビューアは Node.js インストール後に利用可能になります
                  </small>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* README Modal */}
      {showReadme && (
        <div className="modal-overlay" onClick={() => setShowReadme(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">📖 エンジンマニュアル</span>
              <button className="modal-close" onClick={() => setShowReadme(false)}>×</button>
            </div>
            <div className="markdown-content">
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--font-sans)', fontSize: '0.85rem' }}>
                {readmeContent}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-title">⚙ 高度設定</span>
              <button className="modal-close" onClick={() => setShowSettings(false)}>×</button>
            </div>
            <div className="form-group">
              <label className="form-label">HTTP プロキシ</label>
              <input
                className="form-input"
                placeholder="http://proxy.example.com:8080"
                value={proxyHttp}
                onChange={e => setProxyHttp(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">HTTPS プロキシ</label>
              <input
                className="form-input"
                placeholder="http://proxy.example.com:8080"
                value={proxyHttps}
                onChange={e => setProxyHttps(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">HuggingFace Token</label>
              <input
                className="form-input"
                type="password"
                placeholder="hf_xxxxxxxxxxxx"
                value={hfToken}
                onChange={e => setHfToken(e.target.value)}
              />
              <small style={{ color: 'var(--color-text-tertiary)', fontSize: '0.7rem' }}>
                <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" style={{ color: 'var(--color-text-accent)' }}>
                  HuggingFace Token を作成
                </a>
              </small>
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end', marginTop: 'var(--space-lg)' }}>
              <button className="btn btn-secondary" onClick={() => setShowSettings(false)}>キャンセル</button>
              <button className="btn btn-primary" onClick={handleSaveSettings}>保存</button>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' && '✓ '}
          {toast.type === 'error' && '✕ '}
          {toast.type === 'warning' && '⚠ '}
          {toast.msg}
        </div>
      )}
    </div>
  )
}

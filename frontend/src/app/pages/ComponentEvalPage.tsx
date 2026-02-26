import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader2, Cpu, Activity, FileText, Network, Grid3x3, Settings, FolderOpen, Plus, X } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';
import ForceGraph2D from 'react-force-graph-2d';

// ── Top-level dialog components (must NOT be defined inside the parent) ────────

interface QualityConfigDialogProps {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  pplModelPath: string;
  setPplModelPath: (v: string) => void;
  simModelPath: string;
  setSimModelPath: (v: string) => void;
  useVllm: boolean;
  setUseVllm: (v: boolean) => void;
  vllmApiBase: string;
  setVllmApiBase: (v: string) => void;
  vllmModelName: string;
  setVllmModelName: (v: string) => void;
}

function QualityConfigDialog({
  open, onOpenChange,
  pplModelPath, setPplModelPath,
  simModelPath, setSimModelPath,
  useVllm, setUseVllm,
  vllmApiBase, setVllmApiBase,
  vllmModelName, setVllmModelName,
}: QualityConfigDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
          <Settings className="w-4 h-4 text-slate-500 hover:text-blue-600" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Chunk Quality Model Config</DialogTitle>
          <DialogDescription>
            Override default model paths. Leave blank to use server defaults.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div>
            <Label>PPL Model Path</Label>
            <Input
              value={pplModelPath}
              onChange={(e) => setPplModelPath(e.target.value)}
              placeholder="/models/gpt2 (leave blank for default)"
              className="mt-1.5"
            />
            <p className="text-xs text-slate-500 mt-1">Perplexity model for Boundary Clarity</p>
          </div>
          <div>
            <Label>Sim Model Path</Label>
            <Input
              value={simModelPath}
              onChange={(e) => setSimModelPath(e.target.value)}
              placeholder="/models/bge-large-en-v1.5 (leave blank for default)"
              className="mt-1.5"
            />
            <p className="text-xs text-slate-500 mt-1">Embedding model for Semantic Dissimilarity</p>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div>
              <Label className="text-sm">Use vLLM for PPL</Label>
              <p className="text-xs text-slate-500">Call vLLM API instead of local model</p>
            </div>
            <Switch checked={useVllm} onCheckedChange={setUseVllm} />
          </div>
          {useVllm && (
            <div className="space-y-3 pl-3 border-l-2 border-blue-200">
              <div>
                <Label>vLLM API Base</Label>
                <Input
                  value={vllmApiBase}
                  onChange={(e) => setVllmApiBase(e.target.value)}
                  placeholder="http://localhost:8005/v1"
                  className="mt-1.5"
                />
              </div>
              <div>
                <Label>vLLM Model Name</Label>
                <Input
                  value={vllmModelName}
                  onChange={(e) => setVllmModelName(e.target.value)}
                  placeholder="e.g. Qwen2.5-7B-Instruct"
                  className="mt-1.5"
                />
              </div>
            </div>
          )}
        </div>
        <div className="flex justify-end pt-2">
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

interface StickinessConfigDialogProps {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  modelPath: string;
  setModelPath: (v: string) => void;
  useVllm: boolean;
  setUseVllm: (v: boolean) => void;
  vllmApiBase: string;
  setVllmApiBase: (v: string) => void;
  vllmModelName: string;
  setVllmModelName: (v: string) => void;
}

function StickinessConfigDialog({
  open, onOpenChange,
  modelPath, setModelPath,
  useVllm, setUseVllm,
  vllmApiBase, setVllmApiBase,
  vllmModelName, setVllmModelName,
}: StickinessConfigDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
          <Settings className="w-4 h-4 text-slate-500 hover:text-orange-600" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Chunk Stickiness Model Config</DialogTitle>
          <DialogDescription>
            Override default model path. Leave blank to use server default.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div>
            <Label>Model Path</Label>
            <Input
              value={modelPath}
              onChange={(e) => setModelPath(e.target.value)}
              placeholder="/models/bge-large-en-v1.5 (leave blank for default)"
              className="mt-1.5"
            />
            <p className="text-xs text-slate-500 mt-1">Embedding model for structural-entropy evaluation</p>
          </div>
          <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div>
              <Label className="text-sm">Use vLLM for PPL</Label>
              <p className="text-xs text-slate-500">Call vLLM API instead of local model</p>
            </div>
            <Switch checked={useVllm} onCheckedChange={setUseVllm} />
          </div>
          {useVllm && (
            <div className="space-y-3 pl-3 border-l-2 border-orange-200">
              <div>
                <Label>vLLM API Base</Label>
                <Input
                  value={vllmApiBase}
                  onChange={(e) => setVllmApiBase(e.target.value)}
                  placeholder="http://localhost:8005/v1"
                  className="mt-1.5"
                />
              </div>
              <div>
                <Label>vLLM Model Name</Label>
                <Input
                  value={vllmModelName}
                  onChange={(e) => setVllmModelName(e.target.value)}
                  placeholder="e.g. Qwen2.5-7B-Instruct"
                  className="mt-1.5"
                />
              </div>
            </div>
          )}
        </div>
        <div className="flex justify-end pt-2">
          <Button onClick={() => onOpenChange(false)}>Done</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function ComponentEvalPage() {
  const [loading, setLoading] = useState(false);

  // ── Chunk Quality – Direct Input ────────────────────────────────────────────
  const [qualityChunksJson, setQualityChunksJson] = useState('');
  const [enableSemanticSimilarity, setEnableSemanticSimilarity] = useState(true);
  const [enableBoundaryClarity, setEnableBoundaryClarity] = useState(true);
  const [qualityResult, setQualityResult] = useState<any>(null);

  // ── Chunk Quality – File Input ──────────────────────────────────────────────
  const [tempQualityPath, setTempQualityPath] = useState('');
  const [qualityFilePaths, setQualityFilePaths] = useState<string[]>([]);
  const qualityFileRef = useRef<HTMLInputElement>(null);

  // ── Chunk Quality – Model Config (shared) ──────────────────────────────────
  const [qualityConfigOpen, setQualityConfigOpen] = useState(false);
  const [pplModelPath, setPplModelPath] = useState('');
  const [simModelPath, setSimModelPath] = useState('');
  const [qualityUseVllm, setQualityUseVllm] = useState(false);
  const [qualityVllmApiBase, setQualityVllmApiBase] = useState('http://localhost:8005/v1');
  const [qualityVllmModelName, setQualityVllmModelName] = useState('');

  // ── Chunk Stickiness – Direct Input ────────────────────────────────────────
  const [stickinessChunksJson, setStickinessChunksJson] = useState('');
  const [threshold, setThreshold] = useState([0.8]);
  const [delta, setDelta] = useState([0.0]);
  const [stickinessResult, setStickinessResult] = useState<any>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState([0.8]);

  // ── Chunk Stickiness – File Input ──────────────────────────────────────────
  const [tempStickinessPath, setTempStickinessPath] = useState('');
  const [stickinessFilePaths, setStickinessFilePaths] = useState<string[]>([]);
  const stickinessFileRef = useRef<HTMLInputElement>(null);

  // ── Chunk Stickiness – Model Config (shared) ───────────────────────────────
  const [stickinessConfigOpen, setStickinessConfigOpen] = useState(false);
  const [stickinessModelPath, setStickinessModelPath] = useState('');
  const [stickinessUseVllm, setStickinessUseVllm] = useState(false);
  const [stickinessVllmApiBase, setStickinessVllmApiBase] = useState('http://localhost:8005/v1');
  const [stickinessVllmModelName, setStickinessVllmModelName] = useState('');

  // ── Force graph container sizing ───────────────────────────────────────────
  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [graphSize, setGraphSize] = useState({ width: 460, height: 480 });

  useEffect(() => {
    if (!graphContainerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setGraphSize({
          width: entry.contentRect.width,
          height: 480,
        });
      }
    });
    observer.observe(graphContainerRef.current);
    return () => observer.disconnect();
  }, []);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  const buildQualityData = useCallback(
    (base: Record<string, any>) => {
      const data: any = { ...base };
      if (pplModelPath.trim()) data.ppl_model_path = pplModelPath.trim();
      if (simModelPath.trim()) data.sim_model_path = simModelPath.trim();
      if (qualityUseVllm) {
        data.use_vllm = true;
        if (qualityVllmApiBase.trim()) data.vllm_api_base = qualityVllmApiBase.trim();
        if (qualityVllmModelName.trim()) data.vllm_model_name = qualityVllmModelName.trim();
      }
      return data;
    },
    [pplModelPath, simModelPath, qualityUseVllm, qualityVllmApiBase, qualityVllmModelName],
  );

  const buildStickinessData = useCallback(
    (base: Record<string, any>) => {
      const data: any = { ...base };
      if (stickinessModelPath.trim()) data.model_path = stickinessModelPath.trim();
      if (stickinessUseVllm) {
        data.use_vllm = true;
        if (stickinessVllmApiBase.trim()) data.vllm_api_base = stickinessVllmApiBase.trim();
        if (stickinessVllmModelName.trim()) data.vllm_model_name = stickinessVllmModelName.trim();
      }
      return data;
    },
    [stickinessModelPath, stickinessUseVllm, stickinessVllmApiBase, stickinessVllmModelName],
  );

  // ── Handlers ────────────────────────────────────────────────────────────────
  const handleQualityEval = async () => {
    if (!qualityChunksJson.trim()) {
      toast.error('Please enter chunks in JSON format');
      return;
    }
    setLoading(true);
    try {
      const chunks = JSON.parse(qualityChunksJson);
      const data = buildQualityData({
        chunks,
        enable_semantic_similarity: enableSemanticSimilarity,
        enable_boundary_clarity: enableBoundaryClarity,
      });
      const response = await api.chunkQuality(data);
      if (response.success) {
        setQualityResult(response.data);
        toast.success('Chunk quality evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error('Invalid JSON format: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleQualityFileEval = async () => {
    if (!qualityFilePaths.length) {
      toast.error('Please add at least one file path');
      return;
    }
    setLoading(true);
    try {
      const data = buildQualityData({
        input_path: qualityFilePaths[0],
        enable_semantic_similarity: enableSemanticSimilarity,
        enable_boundary_clarity: enableBoundaryClarity,
      });
      const response = await api.chunkQualityFile(data);
      if (response.success) {
        setQualityResult(response.data);
        toast.success('Chunk quality evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStickinessEval = async () => {
    if (!stickinessChunksJson.trim()) {
      toast.error('Please enter chunks in JSON format');
      return;
    }
    setLoading(true);
    try {
      const chunks = JSON.parse(stickinessChunksJson);
      const data = buildStickinessData({ chunks, threshold: threshold[0], delta: delta[0] });
      const response = await api.chunkStickiness(data);
      if (response.success) {
        setStickinessResult(response.data);
        toast.success('Stickiness evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error('Invalid JSON format: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleStickinessFileEval = async () => {
    if (!stickinessFilePaths.length) {
      toast.error('Please add at least one file path');
      return;
    }
    setLoading(true);
    try {
      const data = buildStickinessData({
        input_path: stickinessFilePaths[0],
        threshold: threshold[0],
        delta: delta[0],
      });
      const response = await api.chunkStickinessFile(data);
      if (response.success) {
        setStickinessResult(response.data);
        toast.success('Stickiness evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Auto re-evaluate on param change（仅在非 loading 状态下触发）
  useEffect(() => {
    if (!stickinessChunksJson || !stickinessResult || loading) return;
    const t = setTimeout(() => handleStickinessEval(), 500);
    return () => clearTimeout(t);
  }, [threshold, delta, loading, stickinessChunksJson, stickinessResult]);

  // ── Derived data ────────────────────────────────────────────────────────────
  const qualityChartData =
    qualityResult?.details?.map((item: any, index: number) => ({
      name: `Chunk ${index}`,
      'Boundary Clarity': item.boundary_clarity,
      'Semantic Dissimilarity': item.semantic_dissimilarity,
    })) || [];

  const forceGraphData = useMemo(() => {
    if (!stickinessResult?.graph_incomplete) return null;
    const nodes: any[] = [];
    const links: any[] = [];
    const graph = stickinessResult.graph_incomplete;
    const nodeIds = Object.keys(graph);
    nodeIds.forEach((id) => nodes.push({ id, name: `Chunk ${id}` }));
    nodeIds.forEach((source) => {
      Object.entries(graph[source] || {}).forEach(([target, weight]: [string, any]) => {
        const similarity = 1 - weight;
        if (similarity > similarityThreshold[0]) {
          links.push({ source, target, similarity, value: similarity });
        }
      });
    });
    return { nodes, links };
  }, [stickinessResult, similarityThreshold]);

  const heatmapData = useMemo(() => {
    if (!stickinessResult?.graph_complete) return [];
    const graph = stickinessResult.graph_complete;
    const nodeIds = Object.keys(graph).sort((a, b) => parseInt(a) - parseInt(b));
    const data: any[] = [];
    nodeIds.forEach((i) => {
      nodeIds.forEach((j) => {
        const weight = graph[i]?.[j] ?? 1;
        data.push({ x: parseInt(i), y: parseInt(j), similarity: Math.max(0, 1 - weight) });
      });
    });
    return data;
  }, [stickinessResult]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Component-Level Evaluation</h1>
          <p className="text-slate-600">Evaluate chunk quality and stickiness with advanced visualizations</p>
        </div>

        <Tabs defaultValue="quality">
          <TabsList className="mb-6">
            <TabsTrigger value="quality">Chunk Quality</TabsTrigger>
            <TabsTrigger value="stickiness">Chunk Stickiness</TabsTrigger>
          </TabsList>

          {/* ── Chunk Quality Tab ─────────────────────────────────────────── */}
          <TabsContent value="quality">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input column */}
              <div className="space-y-6">
                {/* Direct Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      Direct Input
                    </h2>
                    <QualityConfigDialog
                      open={qualityConfigOpen}
                      onOpenChange={setQualityConfigOpen}
                      pplModelPath={pplModelPath}
                      setPplModelPath={setPplModelPath}
                      simModelPath={simModelPath}
                      setSimModelPath={setSimModelPath}
                      useVllm={qualityUseVllm}
                      setUseVllm={setQualityUseVllm}
                      vllmApiBase={qualityVllmApiBase}
                      setVllmApiBase={setQualityVllmApiBase}
                      vllmModelName={qualityVllmModelName}
                      setVllmModelName={setQualityVllmModelName}
                    />
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Text (JSON format)</Label>
                      <Textarea
                        value={qualityChunksJson}
                        onChange={(e) => setQualityChunksJson(e.target.value)}
                        placeholder='["First chunk content", "Second chunk content", "Third chunk content"]'
                        className="min-h-[150px] font-mono text-sm"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div>
                          <Label className="text-sm">Semantic Similarity</Label>
                          <p className="text-xs text-slate-500">Cosine distance</p>
                        </div>
                        <Switch checked={enableSemanticSimilarity} onCheckedChange={setEnableSemanticSimilarity} />
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div>
                          <Label className="text-sm">Boundary Clarity</Label>
                          <p className="text-xs text-slate-500">Perplexity ratio</p>
                        </div>
                        <Switch checked={enableBoundaryClarity} onCheckedChange={setEnableBoundaryClarity} />
                      </div>
                    </div>
                    <Button onClick={handleQualityEval} disabled={loading} className="w-full">
                      {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</> : 'Evaluate Quality'}
                    </Button>
                  </div>
                </Card>

                {/* File Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      File Input
                    </h2>
                    <QualityConfigDialog
                      open={qualityConfigOpen}
                      onOpenChange={setQualityConfigOpen}
                      pplModelPath={pplModelPath}
                      setPplModelPath={setPplModelPath}
                      simModelPath={simModelPath}
                      setSimModelPath={setSimModelPath}
                      useVllm={qualityUseVllm}
                      setUseVllm={setQualityUseVllm}
                      vllmApiBase={qualityVllmApiBase}
                      setVllmApiBase={setQualityVllmApiBase}
                      vllmModelName={qualityVllmModelName}
                      setVllmModelName={setQualityVllmModelName}
                    />
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Results File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempQualityPath}
                          onChange={(e) => setTempQualityPath(e.target.value)}
                          placeholder="/path/to/chunks.json or click + to browse"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempQualityPath.trim()) {
                              setQualityFilePaths([...qualityFilePaths, tempQualityPath.trim()]);
                              setTempQualityPath('');
                            }
                          }}
                        />
                        <input
                          type="file"
                          ref={qualityFileRef}
                          accept=".json"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              setQualityFilePaths([...qualityFilePaths, file.name]);
                              if (qualityFileRef.current) qualityFileRef.current.value = '';
                            }
                          }}
                        />
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            if (tempQualityPath.trim()) {
                              setQualityFilePaths([...qualityFilePaths, tempQualityPath.trim()]);
                              setTempQualityPath('');
                            } else {
                              qualityFileRef.current?.click();
                            }
                          }}
                        >
                          {tempQualityPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                        </Button>
                      </div>
                      {qualityFilePaths.length > 0 && (
                        <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                          <div className="p-2 space-y-1">
                            {qualityFilePaths.map((path, idx) => (
                              <div key={idx} className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200">
                                <span className="text-xs font-mono truncate flex-1">{path}</span>
                                <Button
                                  type="button"
                                  onClick={() => setQualityFilePaths(qualityFilePaths.filter((_, i) => i !== idx))}
                                  variant="ghost"
                                  size="sm"
                                  className="h-5 w-5 p-0 text-slate-400 hover:text-red-600"
                                >
                                  <X className="w-3 h-3" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div>
                          <Label className="text-sm">Semantic Similarity</Label>
                          <p className="text-xs text-slate-500">Cosine distance</p>
                        </div>
                        <Switch checked={enableSemanticSimilarity} onCheckedChange={setEnableSemanticSimilarity} />
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div>
                          <Label className="text-sm">Boundary Clarity</Label>
                          <p className="text-xs text-slate-500">Perplexity ratio</p>
                        </div>
                        <Switch checked={enableBoundaryClarity} onCheckedChange={setEnableBoundaryClarity} />
                      </div>
                    </div>
                    <Button onClick={handleQualityFileEval} disabled={loading} className="w-full" variant="outline">
                      {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</> : 'Evaluate from File'}
                    </Button>
                  </div>
                </Card>
              </div>

              {/* Results column */}
              <Card className="p-6">
                <h2 className="font-bold mb-4">Quality Results</h2>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : qualityResult ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-6">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
                          <div className="text-sm text-slate-600 mb-1">Avg Boundary Clarity</div>
                          <div className="text-2xl font-bold text-blue-900">
                            {qualityResult.avg_boundary_clarity?.toFixed(4) || '-'}
                          </div>
                        </div>
                        <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                          <div className="text-sm text-slate-600 mb-1">Avg Semantic Dissimilarity</div>
                          <div className="text-2xl font-bold text-purple-900">
                            {qualityResult.avg_semantic_dissimilarity?.toFixed(4) || '-'}
                          </div>
                        </div>
                      </div>
                      {qualityChartData.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium mb-3">Trends</h3>
                          <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={qualityChartData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Line type="monotone" dataKey="Boundary Clarity" stroke="#3b82f6" strokeWidth={2} />
                              <Line type="monotone" dataKey="Semantic Dissimilarity" stroke="#a855f7" strokeWidth={2} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                      {qualityResult.details?.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium mb-3">Adjacent Chunk Details</h3>
                          <div className="space-y-2">
                            {qualityResult.details.map((detail: any, index: number) => (
                              <div key={index} className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                                <div className="flex items-center justify-between text-sm">
                                  <span className="font-medium">Chunk {index} → {index + 1}</span>
                                  <div className="flex gap-4">
                                    <span className="text-blue-600">BC: {detail.boundary_clarity?.toFixed(4)}</span>
                                    <span className="text-purple-600">Dissim: {detail.semantic_dissimilarity?.toFixed(4)}</span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div>
                        <h3 className="text-sm font-medium mb-2">Complete Results</h3>
                        <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto">
                          {JSON.stringify(qualityResult, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-slate-400">
                    <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Evaluation results will appear here</p>
                  </div>
                )}
              </Card>
            </div>
          </TabsContent>

          {/* ── Chunk Stickiness Tab ──────────────────────────────────────── */}
          <TabsContent value="stickiness">
            <div className="space-y-6">
              {/* Input cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Direct Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <Cpu className="w-5 h-5" />
                      Direct Input
                    </h2>
                    <StickinessConfigDialog
                      open={stickinessConfigOpen}
                      onOpenChange={setStickinessConfigOpen}
                      modelPath={stickinessModelPath}
                      setModelPath={setStickinessModelPath}
                      useVllm={stickinessUseVllm}
                      setUseVllm={setStickinessUseVllm}
                      vllmApiBase={stickinessVllmApiBase}
                      setVllmApiBase={setStickinessVllmApiBase}
                      vllmModelName={stickinessVllmModelName}
                      setVllmModelName={setStickinessVllmModelName}
                    />
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Text (JSON format)</Label>
                      <Textarea
                        value={stickinessChunksJson}
                        onChange={(e) => setStickinessChunksJson(e.target.value)}
                        placeholder='["First chunk content", "Second chunk content", "Third chunk content"]'
                        className="min-h-[150px] font-mono text-sm"
                      />
                    </div>
                    <Button onClick={handleStickinessEval} disabled={loading} className="w-full">
                      {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</> : 'Evaluate Stickiness'}
                    </Button>
                  </div>
                </Card>

                {/* File Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      File Input
                    </h2>
                    <StickinessConfigDialog
                      open={stickinessConfigOpen}
                      onOpenChange={setStickinessConfigOpen}
                      modelPath={stickinessModelPath}
                      setModelPath={setStickinessModelPath}
                      useVllm={stickinessUseVllm}
                      setUseVllm={setStickinessUseVllm}
                      vllmApiBase={stickinessVllmApiBase}
                      setVllmApiBase={setStickinessVllmApiBase}
                      vllmModelName={stickinessVllmModelName}
                      setVllmModelName={setStickinessVllmModelName}
                    />
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Results File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempStickinessPath}
                          onChange={(e) => setTempStickinessPath(e.target.value)}
                          placeholder="/path/to/chunks.json or click + to browse"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempStickinessPath.trim()) {
                              setStickinessFilePaths([...stickinessFilePaths, tempStickinessPath.trim()]);
                              setTempStickinessPath('');
                            }
                          }}
                        />
                        <input
                          type="file"
                          ref={stickinessFileRef}
                          accept=".json"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              setStickinessFilePaths([...stickinessFilePaths, file.name]);
                              if (stickinessFileRef.current) stickinessFileRef.current.value = '';
                            }
                          }}
                        />
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            if (tempStickinessPath.trim()) {
                              setStickinessFilePaths([...stickinessFilePaths, tempStickinessPath.trim()]);
                              setTempStickinessPath('');
                            } else {
                              stickinessFileRef.current?.click();
                            }
                          }}
                        >
                          {tempStickinessPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                        </Button>
                      </div>
                      {stickinessFilePaths.length > 0 && (
                        <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                          <div className="p-2 space-y-1">
                            {stickinessFilePaths.map((path, idx) => (
                              <div key={idx} className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200">
                                <span className="text-xs font-mono truncate flex-1">{path}</span>
                                <Button
                                  type="button"
                                  onClick={() => setStickinessFilePaths(stickinessFilePaths.filter((_, i) => i !== idx))}
                                  variant="ghost"
                                  size="sm"
                                  className="h-5 w-5 p-0 text-slate-400 hover:text-red-600"
                                >
                                  <X className="w-3 h-3" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      )}
                    </div>
                    <Button onClick={handleStickinessFileEval} disabled={loading} className="w-full" variant="outline">
                      {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</> : 'Evaluate from File'}
                    </Button>
                  </div>
                </Card>
              </div>

              {/* Parameter Control Panel */}
              {stickinessResult && (
                <Card className="p-6">
                  <h2 className="font-bold mb-4">Parameter Control</h2>
                  <p className="text-sm text-slate-600 mb-4">
                    Adjust parameters to see their impact on graph structure and entropy
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Threshold</Label>
                        <span className="text-sm font-mono text-slate-600">{threshold[0].toFixed(2)}</span>
                      </div>
                      <Slider
                        value={threshold}
                        onValueChange={setThreshold}
                        min={0}
                        max={1}
                        step={0.01}
                        disabled={loading}
                      />
                      <p className="text-xs text-slate-500">Controls which edges are considered "strong correlation"</p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Delta (Position Penalty)</Label>
                        <span className="text-sm font-mono text-slate-600">{delta[0].toFixed(2)}</span>
                      </div>
                      <Slider
                        value={delta}
                        onValueChange={setDelta}
                        min={0}
                        max={1}
                        step={0.01}
                        disabled={loading}
                      />
                      <p className="text-xs text-slate-500">Higher values favor adjacent chunks</p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Visualization Threshold</Label>
                        <span className="text-sm font-mono text-slate-600">{similarityThreshold[0].toFixed(2)}</span>
                      </div>
                      <Slider
                        value={similarityThreshold}
                        onValueChange={setSimilarityThreshold}
                        min={0}
                        max={1}
                        step={0.01}
                        disabled={loading}
                      />
                      <p className="text-xs text-slate-500">Only show graph edges with similarity above this value</p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Results */}
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                </div>
              ) : stickinessResult ? (
                <>
                  {/* Entropy Values */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="p-2 bg-green-100 rounded-lg">
                          <Cpu className="w-5 h-5 text-green-700" />
                        </div>
                        <div>
                          <h3 className="font-bold">Complete Graph Entropy</h3>
                          <p className="text-xs text-slate-600">All chunks fully connected</p>
                        </div>
                      </div>
                      <div className="text-4xl font-bold text-green-900 mb-2">
                        {stickinessResult.structural_entropy_complete?.toFixed(6) || '-'}
                      </div>
                      <p className="text-sm text-slate-600">Lower values indicate stronger overall cohesion</p>
                    </Card>
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="p-2 bg-orange-100 rounded-lg">
                          <Cpu className="w-5 h-5 text-orange-700" />
                        </div>
                        <div>
                          <h3 className="font-bold">Incomplete Graph Entropy</h3>
                          <p className="text-xs text-slate-600">After threshold filtering</p>
                        </div>
                      </div>
                      <div className="text-4xl font-bold text-orange-900 mb-2">
                        {stickinessResult.structural_entropy_incomplete?.toFixed(6) || '-'}
                      </div>
                      <p className="text-sm text-slate-600">Difference from complete shows threshold impact</p>
                    </Card>
                  </div>

                  {/* Visualizations */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    {/* Force Graph – explicit size to prevent overflow */}
                    <Card className="p-6 overflow-hidden">
                      <div className="flex items-center gap-2 mb-4">
                        <Network className="w-5 h-5" />
                        <h3 className="font-bold">Chunk Dependency Graph</h3>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Force-directed layout — shows edges with similarity above visualization threshold
                      </p>
                      <div
                        ref={graphContainerRef}
                        className="rounded-lg border border-slate-200 bg-slate-50 overflow-hidden"
                        style={{ height: 480 }}
                      >
                        {forceGraphData && forceGraphData.nodes.length > 0 ? (
                          <ForceGraph2D
                            graphData={forceGraphData}
                            width={graphSize.width}
                            height={graphSize.height}
                            nodeLabel="name"
                            nodeAutoColorBy="id"
                            linkWidth={(link: any) => Math.max(1, link.value * 5)}
                            linkColor={() => 'rgba(59, 130, 246, 0.6)'}
                            nodeRelSize={6}
                            linkDirectionalParticles={2}
                            linkDirectionalParticleWidth={(link: any) => link.value * 4}
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center">
                            <div className="text-center text-slate-400">
                              <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
                              <p>No edges above similarity threshold</p>
                              <p className="text-xs">Try lowering the visualization threshold</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </Card>

                    {/* Heatmap */}
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Grid3x3 className="w-5 h-5" />
                        <h3 className="font-bold">Similarity Heatmap</h3>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Matrix view of pairwise chunk similarities (complete graph)
                      </p>
                      {heatmapData.length > 0 ? (
                        <ScrollArea className="h-[480px]">
                          <div
                            className="grid gap-1"
                            style={{
                              gridTemplateColumns: `repeat(${Math.round(Math.sqrt(heatmapData.length))}, minmax(0, 1fr))`,
                            }}
                          >
                            {heatmapData.map((cell, idx) => (
                              <div
                                key={idx}
                                className="aspect-square rounded-sm border border-slate-200"
                                style={{ backgroundColor: `rgba(59, 130, 246, ${cell.similarity})` }}
                                title={`Chunk ${cell.x} → ${cell.y}: ${cell.similarity.toFixed(3)}`}
                              />
                            ))}
                          </div>
                          <div className="flex items-center justify-between mt-4 text-xs">
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-blue-100 rounded border border-slate-200" />
                              Low similarity
                            </span>
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-blue-600 rounded border border-slate-200" />
                              High similarity
                            </span>
                          </div>
                        </ScrollArea>
                      ) : (
                        <div className="text-center py-12 text-slate-400">
                          <Grid3x3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p>No heatmap data available</p>
                        </div>
                      )}
                    </Card>
                  </div>

                  {/* Raw JSON details */}
                  <Card className="p-6 mt-6">
                    <h3 className="text-sm font-medium mb-2">Complete Stickiness Results</h3>
                    <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto max-h-[360px]">
                      {JSON.stringify(stickinessResult, null, 2)}
                    </pre>
                  </Card>
                </>
              ) : null}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
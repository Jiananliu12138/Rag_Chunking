import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { ScrollArea } from '../components/ui/scroll-area';
import { Progress } from '../components/ui/progress';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { Loader2, FlaskConical, Settings, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '../utils/api';
import { DEFAULT_LLM_API_BASE, DEFAULT_LLM_MODEL_NAME, DEFAULT_EMBEDDING_MODEL_PATH } from '../utils/runtimeDefaults';
import { toast } from 'sonner';
import { PathPickerButton } from '../components/PathPickerButton';

interface TestsetResult {
  output_file: string;
  failed_file: string | null;
  total_generated: number;
  total_failed: number;
  testset_size_requested: number;
  samples_preview: any[];
}

interface TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  stage: string;
  progress_pct: number;
  message: string;
  result: TestsetResult | null;
  error: string | null;
}

const STAGE_LABELS: Record<string, string> = {
  queued: 'Queued',
  preflight: 'Checking LLM',
  loading_chunks: 'Loading Chunks',
  building_models: 'Building Models',
  building_transforms: 'Building Transforms',
  generating: 'Generating QA Pairs',
  processing: 'Processing Results',
  writing: 'Writing Output',
  completed: 'Completed',
  failed: 'Failed',
};

export default function TestsetPage() {
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Required
  const [chunksFile, setChunksFile] = useState('');
  const [outputFile, setOutputFile] = useState('');
  const [testsetSize, setTestsetSize] = useState(10);

  // Advanced
  const [maxWorkers, setMaxWorkers] = useState(18);
  const [maxRetries, setMaxRetries] = useState(3);
  const [runTimeout, setRunTimeout] = useState(3600);
  const [failFast, setFailFast] = useState(false);
  const [debugLogs, setDebugLogs] = useState(false);
  const [llmApiBase, setLlmApiBase] = useState(DEFAULT_LLM_API_BASE);
  const [llmModel, setLlmModel] = useState(DEFAULT_LLM_MODEL_NAME);
  const [embeddingPath, setEmbeddingPath] = useState(DEFAULT_EMBEDDING_MODEL_PATH);

  // Task progress
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Result
  const [result, setResult] = useState<TestsetResult | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [stopPolling]);

  const pollTask = useCallback((id: string) => {
    pollingRef.current = setInterval(async () => {
      try {
        const res = await api.getTestsetTaskStatus(id);
        if (!res.success || !res.data) return;
        const status = res.data as TaskStatus;
        setTaskStatus(status);

        if (status.status === 'completed') {
          stopPolling();
          setLoading(false);
          if (status.result) {
            setResult(status.result);
            toast.success(`Generated ${status.result.total_generated} QA pairs`);
          }
        } else if (status.status === 'failed') {
          stopPolling();
          setLoading(false);
          toast.error(status.error || 'Generation failed');
        }
      } catch {
        // Network blip — keep polling
      }
    }, 2000);
  }, [stopPolling]);

  const handleGenerate = async () => {
    if (!chunksFile.trim()) {
      toast.error('Please select a chunks file');
      return;
    }
    setLoading(true);
    setResult(null);
    setTaskStatus(null);
    setTaskId(null);
    stopPolling();

    try {
      const payload: any = {
        chunks_file: chunksFile,
        testset_size: testsetSize,
        max_workers: maxWorkers,
        max_retries: maxRetries,
        run_timeout_seconds: runTimeout,
        fail_fast: failFast,
        with_debugging_logs: debugLogs,
        llm_api_base: llmApiBase || undefined,
        llm_model: llmModel || undefined,
        embedding_model_path: embeddingPath || undefined,
      };
      if (outputFile.trim()) {
        payload.output_file = outputFile;
      }

      const res = await api.generateTestset(payload);
      if (res.success && res.data) {
        const { task_id } = res.data as { task_id: string };
        setTaskId(task_id);
        setTaskStatus({
          task_id,
          status: 'pending',
          stage: 'queued',
          progress_pct: 0,
          message: 'Task submitted',
          result: null,
          error: null,
        });
        pollTask(task_id);
      } else {
        toast.error(res.message || 'Failed to submit task');
        setLoading(false);
      }
    } catch (err: any) {
      toast.error(err.message || 'Request failed');
      setLoading(false);
    }
  };

  const isRunning = loading && taskStatus && !['completed', 'failed'].includes(taskStatus.status);

  return (
    <ScrollArea className="flex-1">
      <div className="p-6 max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FlaskConical className="w-6 h-6 text-purple-600" />
            Testset Generation
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Generate QA test pairs from chunk files using RAGAS TestsetGenerator
          </p>
        </div>

        {/* Config Card */}
        <Card className="p-6 space-y-5">
          {/* Chunks file */}
          <div className="space-y-2">
            <Label>Chunks File *</Label>
            <div className="flex gap-2">
              <Input
                value={chunksFile}
                onChange={e => setChunksFile(e.target.value)}
                placeholder="/path/to/chunks.json"
                className="flex-1"
                disabled={loading}
              />
              <PathPickerButton
                mode="file"
                value={chunksFile}
                onSelect={setChunksFile}
                allowedExtensions={['.json', '.jsonl', '.txt']}
                buttonVariant="outline"
                buttonSize="default"
              />
            </div>
          </div>

          {/* Output file */}
          <div className="space-y-2">
            <Label>Output File (optional)</Label>
            <div className="flex gap-2">
              <Input
                value={outputFile}
                onChange={e => setOutputFile(e.target.value)}
                placeholder="Auto-derived from input filename"
                className="flex-1"
                disabled={loading}
              />
              <PathPickerButton
                mode="directory"
                value={outputFile}
                onSelect={(dir) => setOutputFile(dir + '/testset.jsonl')}
                buttonVariant="outline"
                buttonSize="default"
              />
            </div>
          </div>

          {/* Testset size */}
          <div className="space-y-2">
            <Label>Testset Size</Label>
            <Input
              type="number"
              min={1}
              value={testsetSize}
              onChange={e => setTestsetSize(Number(e.target.value))}
              className="w-32"
              disabled={loading}
            />
          </div>

          {/* Advanced toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-muted-foreground"
          >
            <Settings className="w-4 h-4 mr-1" />
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </Button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-4 p-4 bg-slate-50 rounded-lg">
              <div className="space-y-2">
                <Label>LLM API Base</Label>
                <Input value={llmApiBase} onChange={e => setLlmApiBase(e.target.value)} disabled={loading} />
              </div>
              <div className="space-y-2">
                <Label>LLM Model</Label>
                <Input value={llmModel} onChange={e => setLlmModel(e.target.value)} disabled={loading} />
              </div>
              <div className="space-y-2">
                <Label>Embedding Model Path</Label>
                <Input value={embeddingPath} onChange={e => setEmbeddingPath(e.target.value)} disabled={loading} />
              </div>
              <div className="space-y-2">
                <Label>Max Workers</Label>
                <Input type="number" value={maxWorkers} onChange={e => setMaxWorkers(Number(e.target.value))} disabled={loading} />
              </div>
              <div className="space-y-2">
                <Label>Max Retries</Label>
                <Input type="number" value={maxRetries} onChange={e => setMaxRetries(Number(e.target.value))} disabled={loading} />
              </div>
              <div className="space-y-2">
                <Label>Run Timeout (s)</Label>
                <Input type="number" value={runTimeout} onChange={e => setRunTimeout(Number(e.target.value))} disabled={loading} />
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={failFast} onCheckedChange={setFailFast} disabled={loading} />
                <Label>Fail Fast</Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={debugLogs} onCheckedChange={setDebugLogs} disabled={loading} />
                <Label>Debug Logs</Label>
              </div>
            </div>
          )}

          {/* Generate button */}
          <Button
            onClick={handleGenerate}
            disabled={loading || !chunksFile.trim()}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <FlaskConical className="w-4 h-4 mr-2" />
                Generate Testset
              </>
            )}
          </Button>
        </Card>

        {/* Progress Card */}
        {taskStatus && isRunning && (
          <Card className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                Generating...
              </h2>
              <span className="text-sm font-mono text-muted-foreground">
                {taskStatus.progress_pct}%
              </span>
            </div>

            <Progress value={taskStatus.progress_pct} />

            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-blue-700">
                {STAGE_LABELS[taskStatus.stage] || taskStatus.stage}
              </span>
              <span className="text-muted-foreground">
                {taskStatus.message}
              </span>
            </div>
          </Card>
        )}

        {/* Failed status */}
        {taskStatus?.status === 'failed' && (
          <Card className="p-6 border-red-200 bg-red-50">
            <div className="flex items-center gap-2 text-red-700">
              <XCircle className="w-5 h-5" />
              <h2 className="text-lg font-semibold">Generation Failed</h2>
            </div>
            <p className="mt-2 text-sm text-red-600">{taskStatus.error}</p>
          </Card>
        )}

        {/* Results */}
        {result && (
          <Card className="p-6 space-y-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
              Results
            </h2>

            {/* Summary */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-green-700">{result.total_generated}</div>
                <div className="text-sm text-green-600">Generated</div>
              </div>
              <div className="p-4 bg-red-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-red-700">{result.total_failed}</div>
                <div className="text-sm text-red-600">Failed</div>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-blue-700">{result.testset_size_requested}</div>
                <div className="text-sm text-blue-600">Requested</div>
              </div>
            </div>

            {/* Output path */}
            <div className="text-sm text-muted-foreground">
              <span className="font-medium">Output:</span> {result.output_file}
              {result.failed_file && (
                <span className="ml-4 text-red-500">
                  <span className="font-medium">Failed:</span> {result.failed_file}
                </span>
              )}
            </div>

            {/* Preview table */}
            {result.samples_preview && result.samples_preview.length > 0 && (
              <div>
                <h3 className="text-sm font-medium mb-2">Preview (first {result.samples_preview.length})</h3>
                <div className="border rounded-lg overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">#</TableHead>
                        <TableHead>Question</TableHead>
                        <TableHead>Reference Answer</TableHead>
                        <TableHead className="w-24">Type</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.samples_preview.map((sample: any, idx: number) => (
                        <TableRow key={idx}>
                          <TableCell className="font-mono text-xs">{idx + 1}</TableCell>
                          <TableCell className="max-w-xs truncate text-sm">
                            {sample.user_input || sample.question || '-'}
                          </TableCell>
                          <TableCell className="max-w-xs truncate text-sm">
                            {sample.reference || sample.answer || '-'}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {sample.synthesizer_name || '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}
          </Card>
        )}
      </div>
    </ScrollArea>
  );
}

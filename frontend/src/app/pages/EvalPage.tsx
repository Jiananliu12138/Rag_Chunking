import { useState, useRef } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import { Loader2, BarChart3, FileText, Sparkles, Settings, Upload, FolderOpen, Plus, X } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';

export default function EvalPage() {
  const [loading, setLoading] = useState(false);

  // Traditional Eval - Direct Input
  const [testDataJson, setTestDataJson] = useState('');
  const [enableBertScore, setEnableBertScore] = useState(false);
  const [traditionalResult, setTraditionalResult] = useState<any>(null);

  // Traditional Eval - File Input
  const [tempTraditionalPath, setTempTraditionalPath] = useState('');
  const [traditionalFilePaths, setTraditionalFilePaths] = useState<string[]>([]);
  const traditionalFileRef = useRef<HTMLInputElement>(null);

  // RAGAS Eval - Direct Input
  const [ragasDataJson, setRagasDataJson] = useState('');
  const [ragasResult, setRagasResult] = useState<any>(null);

  // RAGAS Eval - File Input
  const [tempRagasPath, setTempRagasPath] = useState('');
  const [ragasFilePaths, setRagasFilePaths] = useState<string[]>([]);
  const ragasFileRef = useRef<HTMLInputElement>(null);

  // RAGAS Configuration (shared between direct input and file input)
  const [ragasConfigOpen, setRagasConfigOpen] = useState(false);
  const [vllmApiBase, setVllmApiBase] = useState('http://localhost:8005/v1');
  const [vllmModelName, setVllmModelName] = useState('Qwen2.5-7B-Instruct');
  const [embeddingModelPath, setEmbeddingModelPath] = useState('/path/to/bge-large-en-v1.5');

  const handleTraditionalEval = async () => {
    if (!testDataJson.trim()) {
      toast.error('Please enter test data in JSON format');
      return;
    }

    setLoading(true);
    try {
      // Parse the JSON input
      const testData = JSON.parse(testDataJson);
      
      const data: any = {
        test: testData,  // Pass as test field to backend
        enable_bert_score: enableBertScore,
      };

      const response = await api.traditionalEval(data);
      if (response.success) {
        setTraditionalResult(response.data);
        toast.success('Evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error('Invalid JSON format: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleTraditionalFileEval = async () => {
    if (!traditionalFilePaths.length) {
      toast.error('Please add at least one file path');
      return;
    }

    setLoading(true);
    try {
      // Use the first file path if only one API call is supported
      const data: any = {
        input_path: traditionalFilePaths[0],  // Backend expects single path
        enable_bert_score: enableBertScore,
      };

      const response = await api.traditionalEvalFile(data);
      if (response.success) {
        setTraditionalResult(response.data);
        toast.success('Evaluation completed');
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRagasFileEval = async () => {
    if (!ragasFilePaths.length) {
      toast.error('Please add at least one file path');
      return;
    }

    setLoading(true);
    try {
      const response = await api.ragasEvalFile({
        input_path: ragasFilePaths[0],  // Backend expects single path
        vllm_api_base: vllmApiBase,
        vllm_model_name: vllmModelName,
        embedding_model_path: embeddingModelPath,
      });

      if (response.success) {
        setRagasResult(response.data);
        toast.success('RAGAS evaluation completed');
      } else {
        toast.error('RAGAS evaluation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRagasEval = async () => {
    if (!ragasDataJson.trim()) {
      toast.error('Please enter RAGAS data in JSON format');
      return;
    }

    setLoading(true);
    try {
      // Parse the JSON input
      const ragasData = JSON.parse(ragasDataJson);
      
      const data: any = {
        test: ragasData,  // Use 'test' field - backend will auto-parse to dataset format
        vllm_api_base: vllmApiBase,
        vllm_model_name: vllmModelName,
        embedding_model_path: embeddingModelPath,
      };

      const response = await api.ragasEval(data);
      if (response.success) {
        setRagasResult(response.data);
        toast.success('RAGAS evaluation completed');
      } else {
        toast.error('RAGAS evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error('Invalid JSON format: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">End-to-End Evaluation</h1>
          <p className="text-slate-600">Evaluate RAG system quality using traditional metrics and RAGAS</p>
        </div>

        <Tabs defaultValue="traditional">
          <TabsList className="mb-6">
            <TabsTrigger value="traditional">Traditional Metrics</TabsTrigger>
            <TabsTrigger value="ragas">RAGAS Evaluation</TabsTrigger>
          </TabsList>

          {/* Traditional Metrics Tab */}
          <TabsContent value="traditional">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Methods */}
              <div className="space-y-6">
                {/* Direct Input */}
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Direct Input Evaluation
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Test Data (JSON Array)</Label>
                      <Textarea
                        value={testDataJson}
                        onChange={(e) => setTestDataJson(e.target.value)}
                        placeholder={'[\n  {\n    "_id": "question_1",\n    "input": "Who is Peter Rosegger?",\n    "llm_ans": "Peter Rosegger was an Austrian writer and poet.",\n    "answers": ["He was an Austrian writer and poet."],\n    "retrieval_list": ["Peter Rosegger (1843-1918) was..."]\n  }\n]'}
                        className="min-h-[280px] font-mono text-xs"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        💡 Paste your JSON array directly. Each item needs: _id, input, llm_ans, answers, retrieval_list (optional)
                      </p>
                    </div>

                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                      <div>
                        <Label className="text-sm">Enable BERTScore</Label>
                        <p className="text-xs text-slate-500">Semantic similarity</p>
                      </div>
                      <Switch
                        checked={enableBertScore}
                        onCheckedChange={setEnableBertScore}
                      />
                    </div>

                    <Button
                      onClick={handleTraditionalEval}
                      disabled={loading}
                      className="w-full"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Start Evaluation'
                      )}
                    </Button>
                  </div>
                </Card>

                {/* File Input */}
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    File Evaluation
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Evaluation Results File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempTraditionalPath}
                          onChange={(e) => setTempTraditionalPath(e.target.value)}
                          placeholder="/path/to/sample_results.json or click + to browse"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempTraditionalPath.trim()) {
                              setTraditionalFilePaths([...traditionalFilePaths, tempTraditionalPath.trim()]);
                              setTempTraditionalPath('');
                            }
                          }}
                        />
                        <input
                          type="file"
                          ref={traditionalFileRef}
                          accept=".json"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              setTraditionalFilePaths([...traditionalFilePaths, file.name]);
                              if (traditionalFileRef.current) traditionalFileRef.current.value = '';
                            }
                          }}
                        />
                        <Button
                          type="button"
                          onClick={() => {
                            if (tempTraditionalPath.trim()) {
                              setTraditionalFilePaths([...traditionalFilePaths, tempTraditionalPath.trim()]);
                              setTempTraditionalPath('');
                            } else {
                              traditionalFileRef.current?.click();
                            }
                          }}
                          size="sm"
                          variant="outline"
                        >
                          {tempTraditionalPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                        </Button>
                      </div>
                      {traditionalFilePaths.length > 0 && (
                        <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                          <div className="p-2 space-y-1">
                            {traditionalFilePaths.map((path, idx) => (
                              <div
                                key={idx}
                                className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200"
                              >
                                <span className="text-xs font-mono truncate flex-1">{path}</span>
                                <Button
                                  type="button"
                                  onClick={() => setTraditionalFilePaths(traditionalFilePaths.filter((_, i) => i !== idx))}
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

                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                      <div>
                        <Label className="text-sm">Enable BERTScore</Label>
                        <p className="text-xs text-slate-500">Semantic similarity</p>
                      </div>
                      <Switch
                        checked={enableBertScore}
                        onCheckedChange={setEnableBertScore}
                      />
                    </div>

                    <Button
                      onClick={handleTraditionalFileEval}
                      disabled={loading}
                      className="w-full"
                      variant="outline"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Evaluate from File'
                      )}
                    </Button>
                  </div>
                </Card>
              </div>

              {/* Results */}
              <Card className="p-6">
                <h2 className="font-bold mb-4">Evaluation Results</h2>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : traditionalResult ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-6">
                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(traditionalResult).map(([key, value]) => {
                          if (typeof value === 'number') {
                            return (
                              <div
                                key={key}
                                className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200"
                              >
                                <div className="text-sm text-slate-600 mb-1">
                                  {key.replace(/_/g, ' ').toUpperCase()}
                                </div>
                                <div className="text-2xl font-bold text-blue-900">
                                  {(value as number).toFixed(4)}
                                </div>
                              </div>
                            );
                          }
                          return null;
                        })}
                      </div>

                      {/* Raw JSON */}
                      <div>
                        <h3 className="text-sm font-medium mb-2">Complete Results</h3>
                        <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto">
                          {JSON.stringify(traditionalResult, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-slate-400">
                    <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Evaluation results will appear here</p>
                  </div>
                )}
              </Card>
            </div>
          </TabsContent>

          {/* RAGAS Tab */}
          <TabsContent value="ragas">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Methods */}
              <div className="space-y-6">
                {/* Direct JSON Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-purple-600" />
                      Direct JSON Input
                    </h2>
                    <Dialog open={ragasConfigOpen} onOpenChange={setRagasConfigOpen}>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <Settings className="w-4 h-4 text-slate-500 hover:text-purple-600" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="sm:max-w-[500px]">
                        <DialogHeader>
                          <DialogTitle>RAGAS Configuration</DialogTitle>
                          <DialogDescription>
                            Configure VLLM API and embedding model settings
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4 py-4">
                          <div>
                            <Label htmlFor="vllm-api-base-2">VLLM API Base URL</Label>
                            <Input
                              id="vllm-api-base-2"
                              value={vllmApiBase}
                              onChange={(e) => setVllmApiBase(e.target.value)}
                              placeholder="http://localhost:8005/v1"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              The base URL for the VLLM API endpoint
                            </p>
                          </div>
                          <div>
                            <Label htmlFor="vllm-model-name-2">VLLM Model Path</Label>
                            <Input
                              id="vllm-model-name-2"
                              value={vllmModelName}
                              onChange={(e) => setVllmModelName(e.target.value)}
                              placeholder="Qwen2.5-7B-Instruct"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              Path to the VLLM model for evaluation
                            </p>
                          </div>
                          <div>
                            <Label htmlFor="embedding-model-path-2">Embedding Model Path</Label>
                            <Input
                              id="embedding-model-path-2"
                              value={embeddingModelPath}
                              onChange={(e) => setEmbeddingModelPath(e.target.value)}
                              placeholder="/path/to/bge-large-en-v1.5"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              Path to the embedding model for semantic evaluation
                            </p>
                          </div>
                        </div>
                        <div className="flex justify-end">
                          <Button onClick={() => setRagasConfigOpen(false)}>
                            Done
                          </Button>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>RAGAS Data (Flexible JSON Format)</Label>
                      <Textarea
                        value={ragasDataJson}
                        onChange={(e) => setRagasDataJson(e.target.value)}
                        placeholder={'Format 1 - Standard RAGAS:\n{\n  "question": ["Q1?", "Q2?"],\n  "answer": ["A1", "A2"],\n  "contexts": [["C1a", "C1b"], ["C2"]],\n  "ground_truth": ["GT1", "GT2"]\n}\n\nFormat 2 - Sample Results:\n[\n  {\n    "_id": "q1",\n    "input": "Question?",\n    "llm_ans": "LLM answer",\n    "answers": ["Ground truth"],\n    "retrieval_list": ["Context 1", "Context 2"]\n  }\n]'}
                        className="min-h-[360px] font-mono text-xs"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        💡 Supports both formats. Backend auto-detects and converts.
                      </p>
                    </div>

                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                      <h3 className="text-sm font-medium mb-2">RAGAS Metrics</h3>
                      <ul className="text-xs text-slate-600 space-y-1">
                        <li>• Faithfulness - Answer faithfulness to contexts</li>
                        <li>• Answer Relevancy - Relevance of answer to question</li>
                        <li>• Context Recall - Recall of retrieved contexts</li>
                        <li>• Context Precision - Accuracy of retrieved contexts</li>
                        <li>• Context Entity Recall - Entity-level coverage</li>
                        <li>• Noise Sensitivity - Robustness to noise</li>
                      </ul>
                    </div>

                    <Button
                      onClick={handleRagasEval}
                      disabled={loading}
                      className="w-full bg-gradient-to-r from-purple-600 to-pink-600"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Start RAGAS Evaluation'
                      )}
                    </Button>
                  </div>
                </Card>

                {/* File Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <FileText className="w-5 h-5 text-purple-600" />
                      File Evaluation
                    </h2>
                    <Dialog open={ragasConfigOpen} onOpenChange={setRagasConfigOpen}>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                          <Settings className="w-4 h-4 text-slate-500 hover:text-purple-600" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="sm:max-w-[500px]">
                        <DialogHeader>
                          <DialogTitle>RAGAS Configuration</DialogTitle>
                          <DialogDescription>
                            Configure VLLM API and embedding model settings
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4 py-4">
                          <div>
                            <Label htmlFor="vllm-api-base-2">VLLM API Base URL</Label>
                            <Input
                              id="vllm-api-base-2"
                              value={vllmApiBase}
                              onChange={(e) => setVllmApiBase(e.target.value)}
                              placeholder="http://localhost:8005/v1"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              The base URL for the VLLM API endpoint
                            </p>
                          </div>
                          <div>
                            <Label htmlFor="vllm-model-name-2">VLLM Model Path</Label>
                            <Input
                              id="vllm-model-name-2"
                              value={vllmModelName}
                              onChange={(e) => setVllmModelName(e.target.value)}
                              placeholder="Qwen2.5-7B-Instruct"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              Path to the VLLM model for evaluation
                            </p>
                          </div>
                          <div>
                            <Label htmlFor="embedding-model-path-2">Embedding Model Path</Label>
                            <Input
                              id="embedding-model-path-2"
                              value={embeddingModelPath}
                              onChange={(e) => setEmbeddingModelPath(e.target.value)}
                              placeholder="/path/to/bge-large-en-v1.5"
                              className="mt-1.5"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              Path to the embedding model for semantic evaluation
                            </p>
                          </div>
                        </div>
                        <div className="flex justify-end">
                          <Button onClick={() => setRagasConfigOpen(false)}>
                            Done
                          </Button>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Evaluation Data File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempRagasPath}
                          onChange={(e) => setTempRagasPath(e.target.value)}
                          placeholder="/path/to/ragas_data.json or click + to browse"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempRagasPath.trim()) {
                              setRagasFilePaths([...ragasFilePaths, tempRagasPath.trim()]);
                              setTempRagasPath('');
                            }
                          }}
                        />
                        <input
                          type="file"
                          ref={ragasFileRef}
                          accept=".json"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                              setRagasFilePaths([...ragasFilePaths, file.name]);
                              if (ragasFileRef.current) ragasFileRef.current.value = '';
                            }
                          }}
                        />
                        <Button
                          type="button"
                          onClick={() => {
                            if (tempRagasPath.trim()) {
                              setRagasFilePaths([...ragasFilePaths, tempRagasPath.trim()]);
                              setTempRagasPath('');
                            } else {
                              ragasFileRef.current?.click();
                            }
                          }}
                          size="sm"
                          variant="outline"
                        >
                          {tempRagasPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                        </Button>
                      </div>
                      {ragasFilePaths.length > 0 && (
                        <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                          <div className="p-2 space-y-1">
                            {ragasFilePaths.map((path, idx) => (
                              <div
                                key={idx}
                                className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200"
                              >
                                <span className="text-xs font-mono truncate flex-1">{path}</span>
                                <Button
                                  type="button"
                                  onClick={() => setRagasFilePaths(ragasFilePaths.filter((_, i) => i !== idx))}
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
                      <p className="text-xs text-slate-500 mt-1">
                        Supports standard RAGAS format or sample_results.json format
                      </p>
                    </div>

                    <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
                      <p className="text-xs text-amber-800">
                        💡 The file will be automatically parsed based on its format
                      </p>
                    </div>

                    <Button
                      onClick={handleRagasFileEval}
                      disabled={loading}
                      className="w-full"
                      variant="outline"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Evaluate from File'
                      )}
                    </Button>
                  </div>
                </Card>
              </div>

              {/* Results */}
              <Card className="p-6">
                <h2 className="font-bold mb-4">RAGAS Results</h2>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : ragasResult ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-6">
                      {/* Summary Metrics */}
                      {ragasResult.summary && (
                        <div>
                          <h3 className="text-sm font-medium mb-3">Summary</h3>
                          <div className="grid grid-cols-2 gap-3">
                            {Object.entries(ragasResult.summary).map(([key, value]) => {
                              // Handle both simple numbers and {mean, min, max} objects
                              let displayValue: string;
                              if (typeof value === 'number') {
                                displayValue = value.toFixed(4);
                              } else if (value && typeof value === 'object' && 'mean' in value) {
                                displayValue = `${(value as any).mean.toFixed(4)}`;
                              } else {
                                displayValue = String(value);
                              }
                              
                              return (
                                <div
                                  key={key}
                                  className="p-3 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200"
                                >
                                  <div className="text-xs text-slate-600 mb-1">
                                    {key.replace(/_/g, ' ').toUpperCase()}
                                  </div>
                                  <div className="text-xl font-bold text-purple-900">
                                    {displayValue}
                                  </div>
                                  {value && typeof value === 'object' && 'mean' in value && (
                                    <div className="text-xs text-slate-500 mt-1">
                                      min: {(value as any).min.toFixed(3)} | max: {(value as any).max.toFixed(3)}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Sample Details */}
                      {ragasResult.samples && ragasResult.samples.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium mb-3">Sample Details</h3>
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>#</TableHead>
                                <TableHead>Question</TableHead>
                                <TableHead>Scores</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {ragasResult.samples.slice(0, 10).map((sample: any, idx: number) => (
                                <TableRow key={idx}>
                                  <TableCell>{idx + 1}</TableCell>
                                  <TableCell className="max-w-xs truncate">
                                    {sample.question || sample.input || '-'}
                                  </TableCell>
                                  <TableCell>
                                    <div className="text-xs space-y-1">
                                      {Object.entries(sample)
                                        .filter(([key]) => key !== 'question' && key !== 'input')
                                        .map(([key, val]) => (
                                          <div key={key}>
                                            {key}: {typeof val === 'number' ? (val as number).toFixed(3) : '-'}
                                          </div>
                                        ))}
                                    </div>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      )}

                      {/* Raw JSON */}
                      <div>
                        <h3 className="text-sm font-medium mb-2">Complete Results</h3>
                        <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto">
                          {JSON.stringify(ragasResult, null, 2)}
                        </pre>
                      </div>
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-slate-400">
                    <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>RAGAS results will appear here</p>
                  </div>
                )}
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
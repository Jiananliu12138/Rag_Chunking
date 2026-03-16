import { useState, useEffect, type KeyboardEvent } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import { Badge } from '../components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import { Loader2, FileText, Type, Scissors, X, Eraser } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';
import { PathPickerButton } from '../components/PathPickerButton';

interface ChunkMethod {
  name: string;
  description: string;
  params_schema: any;
}

export default function ChunkingPage() {
  const [methods, setMethods] = useState<ChunkMethod[]>([]);
  const [selectedMethod, setSelectedMethod] = useState<string>('token');
  const [loading, setLoading] = useState(false);

  // Text chunking state
  const [inputText, setInputText] = useState('');
  const [textChunks, setTextChunks] = useState<string[][]>([]);
  const [textParams, setTextParams] = useState<any>({ chunk_size: 512, chunk_overlap: 50 });

  // File chunking state
  const [inputPath, setInputPath] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [fileResult, setFileResult] = useState<any>(null);
  const [fileParams, setFileParams] = useState<any>({ chunk_size: 512, chunk_overlap: 50 });

  useEffect(() => {
    loadMethods();
  }, []);

  useEffect(() => {
    // Clear chunks when method changes
    setTextChunks([]);
  }, [selectedMethod]);

  const tabFill = (
    setter: (value: string) => void,
  ) => (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key !== 'Tab') return;
    const current = e.currentTarget;
    if (!current.value?.trim() && current.placeholder?.trim()) {
      setter(current.placeholder);
    }
  };

  const loadMethods = async () => {
    const response = await api.getChunkMethods();
    if (response.success) {
      setMethods(response.data);
    } else {
      toast.error('Failed to load chunking methods: ' + response.message);
    }
  };

  const handleTextChunk = async () => {
    if (!inputText.trim()) {
      toast.error('Please enter text to chunk');
      return;
    }

    setLoading(true);
    try {
      const data: any = {
        method: selectedMethod,
        text: inputText,
      };

      // Add method-specific params
      if (selectedMethod === 'token') {
        data.token_params = textParams;
      } else if (selectedMethod === 'semantic') {
        data.semantic_params = textParams;
      } else if (selectedMethod === 'llamaindex') {
        data.llamaindex_params = textParams;
      } else if (selectedMethod === 'lumber') {
        data.lumber_params = textParams;
      }

      const response = await api.chunkText(data);
      if (response.success) {
        setTextChunks(response.data.splits || []);
        toast.success(`Chunking completed! Time: ${response.data.time_cost?.toFixed(2)}s`);
      } else {
        toast.error('Chunking failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileChunk = async () => {
    if (!inputPath || !outputDir) {
      toast.error('Please fill in input path and output directory');
      return;
    }

    setLoading(true);
    try {
      const data: any = {
        method: selectedMethod,
        input_file: inputPath,  // Backend expects input_file, not input_path
        output_dir: outputDir,
      };

      // Add method-specific params
      if (selectedMethod === 'token') {
        data.token_params = fileParams;
      } else if (selectedMethod === 'semantic') {
        data.semantic_params = fileParams;
      } else if (selectedMethod === 'llamaindex') {
        data.llamaindex_params = fileParams;
      } else if (selectedMethod === 'lumber') {
        data.lumber_params = fileParams;
      }

      const response = await api.chunkFile(data);
      if (response.success) {
        setFileResult(response.data);
        toast.success(`File chunking completed!`);
      } else {
        toast.error('File chunking failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const selectedMethodData = methods.find(m => m.name === selectedMethod);

  // Render params based on selected method
  const renderParams = (params: any, setParams: (p: any) => void) => {
    return (
      <>
        {/* Token method params */}
        {selectedMethod === 'token' && (
          <>
            <div>
              <Label>Chunk Token Size</Label>
              <Input
                type="number"
                value={params.chunk_token_size || 1200}
                onChange={(e) => setParams({ ...params, chunk_token_size: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Chunk Overlap Token Size</Label>
              <Input
                type="number"
                value={params.chunk_overlap_token_size || 100}
                onChange={(e) => setParams({ ...params, chunk_overlap_token_size: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Split by Character (Optional)</Label>
              <Input
                value={params.split_by_character || '\\n\\n'}
                onChange={(e) => setParams({ ...params, split_by_character: e.target.value })}
                onKeyDown={tabFill((value) => setParams({ ...params, split_by_character: value }))}
                placeholder="\\n\\n"
              />
            </div>
          </>
        )}

        {/* Semantic method params */}
        {selectedMethod === 'semantic' && (
          <>
            <div>
              <Label>Embed Model Path (Optional)</Label>
              <Input
                value={params.embed_model_path || ''}
                onChange={(e) => setParams({ ...params, embed_model_path: e.target.value })}
                onKeyDown={tabFill((value) => setParams({ ...params, embed_model_path: value }))}
                placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
              />
            </div>
            <div>
              <Label>Buffer Size</Label>
              <Input
                type="number"
                value={params.buffer_size || 1}
                onChange={(e) => setParams({ ...params, buffer_size: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Breakpoint Percentile Threshold</Label>
              <Input
                type="number"
                value={params.breakpoint_percentile_threshold || 74}
                onChange={(e) => setParams({ ...params, breakpoint_percentile_threshold: parseInt(e.target.value) })}
              />
            </div>
          </>
        )}

        {/* LlamaIndex method params */}
        {selectedMethod === 'llamaindex' && (
          <>
            <div>
              <Label>Chunk Size</Label>
              <Input
                type="number"
                value={params.chunk_size || 512}
                onChange={(e) => setParams({ ...params, chunk_size: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={params.chunk_overlap || 50}
                onChange={(e) => setParams({ ...params, chunk_overlap: parseInt(e.target.value) })}
              />
            </div>
          </>
        )}

        {/* Lumber method params */}
        {selectedMethod === 'lumber' && (
          <>
            <div>
              <Label>LLM API Base (Optional)</Label>
              <Input
                value={params.llm_api_base || ''}
                onChange={(e) => setParams({ ...params, llm_api_base: e.target.value })}
                onKeyDown={tabFill((value) => setParams({ ...params, llm_api_base: value }))}
                placeholder="http://localhost:8005/v1"
              />
            </div>
            <div>
              <Label>Model Type (Optional)</Label>
              <Input
                value={params.model_type || ''}
                onChange={(e) => setParams({ ...params, model_type: e.target.value })}
                onKeyDown={tabFill((value) => setParams({ ...params, model_type: value }))}
                placeholder="Qwen2.5-7B-Instruct"
              />
            </div>
            <div>
              <Label>Temperature</Label>
              <Input
                type="number"
                step="0.1"
                value={params.temperature || 0.2}
                onChange={(e) => setParams({ ...params, temperature: parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <Label>Max Tokens</Label>
              <Input
                type="number"
                value={params.max_tokens || 3072}
                onChange={(e) => setParams({ ...params, max_tokens: parseInt(e.target.value) })}
              />
            </div>
          </>
        )}
      </>
    );
  };

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Text Chunking</h1>
          <p className="text-slate-600">Split text into semantic units using various methods</p>
        </div>

        <Tabs defaultValue="text">
          <TabsList className="mb-4">
            <TabsTrigger value="text" className="flex items-center gap-2">
              <Type className="w-4 h-4" />
              Text Chunking
            </TabsTrigger>
            <TabsTrigger value="file" className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              File Chunking
            </TabsTrigger>
          </TabsList>

          {/* Text Chunking Tab */}
          <TabsContent value="text">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-220px)]">
              {/* Input */}
              <Card className="p-6 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold">Input Text</h3>
                  {inputText && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setInputText('')}
                      className="h-8 text-slate-500 hover:text-slate-700"
                    >
                      <Eraser className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  )}
                </div>
                <Textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={tabFill(setInputText)}
                  placeholder="Enter text to chunk here..."
                  className="min-h-[200px] mb-4"
                />

                {/* Method Selection */}
                <div className="space-y-4 mb-4">
                  <div>
                    <Label>Chunking Method</Label>
                    <Select value={selectedMethod} onValueChange={setSelectedMethod}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="token">Token-based</SelectItem>
                        <SelectItem value="semantic">Semantic</SelectItem>
                        <SelectItem value="llamaindex">LlamaIndex</SelectItem>
                        <SelectItem value="lumber">Lumber</SelectItem>
                      </SelectContent>
                    </Select>
                    {selectedMethodData && (
                      <p className="text-xs text-slate-500 mt-1">
                        {selectedMethodData.description}
                      </p>
                    )}
                  </div>

                  {renderParams(textParams, setTextParams)}
                </div>

                <Button
                  onClick={handleTextChunk}
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Chunking...
                    </>
                  ) : (
                    <>
                      <Scissors className="w-4 h-4 mr-2" />
                      Start Chunking
                    </>
                  )}
                </Button>
              </Card>

              {/* Output */}
              <Card className="p-6 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold">Chunking Results</h3>
                  {textChunks.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setTextChunks([])}
                      className="h-8 text-slate-500 hover:text-slate-700"
                    >
                      <X className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  )}
                </div>
                {textChunks.length > 0 ? (
                  <ScrollArea className="flex-1 overflow-auto">
                    <div className="space-y-3 pr-4">
                      {textChunks.map((chunk, index) => (
                        <div
                          key={index}
                          className="p-4 bg-slate-50 rounded-lg border border-slate-200"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <Badge variant="outline">Chunk {index + 1}</Badge>
                            <span className="text-xs text-slate-500">
                              {chunk[0]?.length || 0} chars
                            </span>
                          </div>
                          <p className="text-base leading-relaxed text-slate-700 whitespace-pre-wrap break-words">{chunk[0]}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Scissors className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Chunking results will appear here</p>
                    </div>
                  </div>
                )}
              </Card>
            </div>
          </TabsContent>

          {/* File Chunking Tab */}
          <TabsContent value="file">
            <Card className="p-6 max-w-3xl mx-auto overflow-hidden">
              <h3 className="font-bold mb-4">File Chunking Configuration</h3>
              
              <div className="space-y-4 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Input File Path</Label>
                    <div className="flex gap-2">
                      <Input
                        value={inputPath}
                        onChange={(e) => setInputPath(e.target.value)}
                        onKeyDown={tabFill(setInputPath)}
                        placeholder="/path/to/input.json"
                      />
                      <PathPickerButton
                        mode="file"
                        value={inputPath}
                        title="Select Input File"
                        description="This browser reads the filesystem on the machine running the backend service."
                        onSelect={setInputPath}
                      />
                    </div>
                  </div>
                  <div>
                    <Label>Output Directory</Label>
                    <div className="flex gap-2">
                      <Input
                        value={outputDir}
                        onChange={(e) => setOutputDir(e.target.value)}
                        onKeyDown={tabFill(setOutputDir)}
                        placeholder="/path/to/output"
                      />
                      <PathPickerButton
                        mode="directory"
                        value={outputDir}
                        title="Select Output Directory"
                        description="This browser reads the filesystem on the machine running the backend service."
                        onSelect={setOutputDir}
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <Label>Chunking Method</Label>
                  <Select value={selectedMethod} onValueChange={setSelectedMethod}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="token">Token-based</SelectItem>
                      <SelectItem value="semantic">Semantic</SelectItem>
                      <SelectItem value="llamaindex">LlamaIndex</SelectItem>
                      <SelectItem value="lumber">Lumber</SelectItem>
                    </SelectContent>
                  </Select>
                  {selectedMethodData && (
                    <p className="text-xs text-slate-500 mt-1">
                      {selectedMethodData.description}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {renderParams(fileParams, setFileParams)}
                </div>
              </div>

              <Button
                onClick={handleFileChunk}
                disabled={loading}
                className="w-full mb-6"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4 mr-2" />
                    Start File Chunking
                  </>
                )}
              </Button>

              {fileResult && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h4 className="font-medium mb-2 text-green-900">Chunking Completed</h4>
                  <div className="text-sm space-y-1 text-green-800">
                    <p className="break-all"><strong>Output File:</strong> {fileResult.output_file}</p>
                    <p><strong>Time Cost:</strong> {fileResult.time_cost?.toFixed(2)}s</p>
                  </div>
                </div>
              )}
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

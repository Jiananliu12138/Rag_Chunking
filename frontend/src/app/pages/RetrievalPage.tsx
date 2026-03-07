import { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../components/ui/collapsible';
import { Loader2, Search, Sparkles, FileText, ChevronDown } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';

interface SearchResult {
  text: string;
  score: number;
  filepath?: string;
  doc_id?: string;
}

export default function RetrievalPage() {
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [collectionName, setCollectionName] = useState('');
  const [embedModelPath, setEmbedModelPath] = useState('');
  const [embedDim, setEmbedDim] = useState(768);
  const [topK, setTopK] = useState(5);
  const [filepath, setFilepath] = useState('');
  const [docId, setDocId] = useState('');
  const [useHybridSearch, setUseHybridSearch] = useState(false);

  // RAG params
  const [llmApiBase, setLlmApiBase] = useState('');
  const [llmModelName, setLlmModelName] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [maxNewTokens, setMaxNewTokens] = useState(512);

  // Results
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [ragAnswer, setRagAnswer] = useState('');
  const [ragContexts, setRagContexts] = useState<string[]>([]);
  const [ragContextItems, setRagContextItems] = useState<SearchResult[]>([]);

  // File generation
  const [inputPath, setInputPath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [fileResult, setFileResult] = useState<any>(null);

  const handleSearch = async () => {
    if (!query || !collectionName) {
      toast.error('请填写查询和集合名称');
      return;
    }

    setLoading(true);
    setSearchResults([]);
    try {
      const data: any = {
        query,
        collection_name: collectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
      };

      if (filepath) data.filepath = filepath;
      if (docId) data.doc_id = docId;

      const response = await api.search(data);
      if (response.success) {
        setSearchResults(response.data.results || []);
        toast.success(`找到 ${response.data.results?.length || 0} 条结果`);
      } else {
        toast.error('检索失败: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!query || !collectionName) {
      toast.error('请填写查询和集合名称');
      return;
    }

    setLoading(true);
    setRagAnswer('');
    setRagContexts([]);
    setRagContextItems([]);
    try {
      const data: any = {
        query,
        collection_name: collectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
        llm_api_base: llmApiBase || undefined,
        llm_model_name: llmModelName || undefined,
        temperature,
        max_new_tokens: maxNewTokens,
      };

      if (filepath) data.filepath = filepath;
      if (docId) data.doc_id = docId;

      const response = await api.generate(data);
      if (response.success) {
        setRagAnswer(response.data.answer || '');
        setRagContexts(response.data.contexts || []);
        setRagContextItems(response.data.context_items || []);
        toast.success('生成成功');
      } else {
        toast.error('生成失败: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateFile = async () => {
    if (!inputPath || !outputPath || !collectionName) {
      toast.error('请填写所有必需字段');
      return;
    }

    setLoading(true);
    try {
      const data: any = {
        input_path: inputPath,
        output_path: outputPath,
        collection_name: collectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
        llm_api_base: llmApiBase || undefined,
        llm_model_name: llmModelName || undefined,
        temperature,
        max_new_tokens: maxNewTokens,
      };

      const response = await api.generateFile(data);
      if (response.success) {
        setFileResult(response.data);
        toast.success(`批处理完成！处理了 ${response.data.total_processed} 条`);
      } else {
        toast.error('批处理失败: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">检索与 RAG</h1>
        <p className="text-slate-600">向量检索和检索增强生成</p>
      </div>

      <Tabs defaultValue="single">
        <TabsList className="mb-6">
          <TabsTrigger value="single">单次检索/生成</TabsTrigger>
          <TabsTrigger value="batch">批量处理</TabsTrigger>
        </TabsList>

        {/* Single Query Tab */}
        <TabsContent value="single">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: Configuration */}
            <Card className="p-6 lg:col-span-1">
              <h2 className="font-bold mb-4">配置</h2>
              <ScrollArea className="h-[600px] pr-4">
                <div className="space-y-4">
                  <div>
                    <Label>查询</Label>
                    <Textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="输入查询问题..."
                      className="min-h-[100px]"
                    />
                  </div>

                  <div>
                    <Label>集合名称</Label>
                    <Input
                      value={collectionName}
                      onChange={(e) => setCollectionName(e.target.value)}
                      placeholder="my_collection"
                    />
                  </div>

                  <div>
                    <Label>Embedding 模型路径 (可选)</Label>
                    <Input
                      value={embedModelPath}
                      onChange={(e) => setEmbedModelPath(e.target.value)}
                      placeholder="留空使用默认"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Embed Dim</Label>
                      <Input
                        type="number"
                        value={embedDim}
                        onChange={(e) => setEmbedDim(parseInt(e.target.value))}
                      />
                    </div>
                    <div>
                      <Label>Top K</Label>
                      <Input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                      />
                    </div>
                  </div>

                  <div>
                    <Label>文件路径过滤 (可选)</Label>
                    <Input
                      value={filepath}
                      onChange={(e) => setFilepath(e.target.value)}
                      placeholder="过滤特定文件"
                    />
                  </div>

                  <div>
                    <Label>Doc ID 过滤 (可选)</Label>
                    <Input
                      value={docId}
                      onChange={(e) => setDocId(e.target.value)}
                      placeholder="过滤特定文档"
                    />
                  </div>

                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div>
                      <Label className="text-sm">Hybrid Search</Label>
                      <p className="text-xs text-slate-500">Dense + Sparse</p>
                    </div>
                    <Switch
                      checked={useHybridSearch}
                      onCheckedChange={setUseHybridSearch}
                    />
                  </div>

                  <div className="border-t pt-4 mt-4">
                    <h3 className="font-medium mb-3 text-sm">LLM 配置 (RAG 需要)</h3>
                    <div className="space-y-3">
                      <div>
                        <Label className="text-sm">API Base (可选)</Label>
                        <Input
                          value={llmApiBase}
                          onChange={(e) => setLlmApiBase(e.target.value)}
                          placeholder="http://localhost:8000/v1"
                        />
                      </div>
                      <div>
                        <Label className="text-sm">模型名称 (可选)</Label>
                        <Input
                          value={llmModelName}
                          onChange={(e) => setLlmModelName(e.target.value)}
                          placeholder="gpt-3.5-turbo"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <Label className="text-sm">Temperature</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Max Tokens</Label>
                          <Input
                            type="number"
                            value={maxNewTokens}
                            onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 pt-4">
                    <Button
                      onClick={handleSearch}
                      disabled={loading}
                      variant="outline"
                    >
                      <Search className="w-4 h-4 mr-2" />
                      只检索
                    </Button>
                    <Button
                      onClick={handleGenerate}
                      disabled={loading}
                      className="bg-gradient-to-r from-purple-600 to-pink-600"
                    >
                      <Sparkles className="w-4 h-4 mr-2" />
                      检索+生成
                    </Button>
                  </div>
                </div>
              </ScrollArea>
            </Card>

            {/* Right: Results */}
            <div className="lg:col-span-2 space-y-6">
              {/* Search Results */}
              <Card className="p-6">
                <h2 className="font-bold mb-4 flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  检索结果
                  {useHybridSearch && (
                    <Badge className="bg-blue-600">Hybrid Search</Badge>
                  )}
                </h2>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : searchResults.length > 0 ? (
                  <ScrollArea className="h-[400px]">
                    <div className="space-y-3">
                      {searchResults.map((result, index) => (
                        <div
                          key={index}
                          className="p-4 bg-slate-50 rounded-lg border border-slate-200 hover:border-blue-300 transition-colors"
                        >
                          <div className="flex items-start justify-between mb-2">
                            <Badge variant="outline">#{index + 1}</Badge>
                            <div className="text-right text-sm">
                              <div className="font-medium text-blue-600">
                                Score: {result.score.toFixed(4)}
                              </div>
                              {result.filepath && (
                                <div className="text-xs text-slate-500 mt-1">
                                  {result.filepath}
                                </div>
                              )}
                            </div>
                          </div>
                          <p className="text-sm text-slate-700">{result.text}</p>
                          {result.doc_id && (
                            <div className="text-xs text-slate-500 mt-2">
                              Doc ID: {result.doc_id}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-slate-400">
                    <Search className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>检索结果将显示在这里</p>
                  </div>
                )}
              </Card>

              {/* RAG Answer */}
              {ragAnswer && (
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-purple-600" />
                    RAG 生成答案
                  </h2>
                  <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200 mb-4">
                    <p className="text-slate-800 whitespace-pre-wrap">{ragAnswer}</p>
                  </div>

                  <Collapsible>
                    <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900">
                      <ChevronDown className="w-4 h-4" />
                      查看使用的上下文 ({ragContexts.length} 条)
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-4">
                      <div className="space-y-2">
                        {ragContextItems.map((item, index) => (
                          <div
                            key={index}
                            className="p-3 bg-white rounded border border-slate-200 text-sm"
                          >
                            <div className="flex items-center justify-between mb-1">
                              <Badge variant="outline" className="text-xs">
                                Context {index + 1}
                              </Badge>
                              <span className="text-xs text-slate-500">
                                Score: {item.score.toFixed(4)}
                              </span>
                            </div>
                            <p className="text-slate-700">{item.text}</p>
                          </div>
                        ))}
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Batch Processing Tab */}
        <TabsContent value="batch">
          <Card className="p-6 max-w-4xl mx-auto">
            <h2 className="font-bold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5" />
              批量 RAG 生成
            </h2>
            <div className="space-y-4 mb-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>输入文件路径</Label>
                  <Input
                    value={inputPath}
                    onChange={(e) => setInputPath(e.target.value)}
                    placeholder="/path/to/input.jsonl"
                  />
                </div>
                <div>
                  <Label>输出文件路径</Label>
                  <Input
                    value={outputPath}
                    onChange={(e) => setOutputPath(e.target.value)}
                    placeholder="/path/to/output.json"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>集合名称</Label>
                  <Input
                    value={collectionName}
                    onChange={(e) => setCollectionName(e.target.value)}
                    placeholder="my_collection"
                  />
                </div>
                <div>
                  <Label>Top K</Label>
                  <Input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <Button
              onClick={handleGenerateFile}
              disabled={loading}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  批处理中...
                </>
              ) : (
                '开始批量生成'
              )}
            </Button>

            {fileResult && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-medium mb-2 text-green-900">处理完成</h3>
                <div className="text-sm space-y-1 text-green-800">
                  <p><strong>输出文件:</strong> {fileResult.output_file}</p>
                  <p><strong>处理数量:</strong> {fileResult.total_processed}</p>
                  <p><strong>失败数量:</strong> {fileResult.total_failed}</p>
                  {fileResult.message && <p><strong>消息:</strong> {fileResult.message}</p>}
                </div>
              </div>
            )}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

import { useState, useEffect, useCallback, useMemo, type KeyboardEvent } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Badge } from '../components/ui/badge';
import { Checkbox } from '../components/ui/checkbox';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../components/ui/collapsible';
import { Popover, PopoverContent, PopoverTrigger } from '../components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '../components/ui/command';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Loader2, Search, Sparkles, FileText, ChevronDown, Settings2, Filter, X, ChevronsUpDown } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';
import { PathPickerButton } from '../components/PathPickerButton';

interface SearchResult {
  text: string;
  score: number;
  filepath?: string;
  doc_id?: string;
  chunk_id?: string;
}

const DEFAULT_EMBED_DIM = 1024;
const DEFAULT_TOP_K = 5;
const DEFAULT_RERANK_CANDIDATE_K = 20;
const DEFAULT_RERANK_TOP_K = 5;
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_MAX_NEW_TOKENS = 512;

const parseIntegerInput = (value: string, fallback: number) => {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const parseFloatInput = (value: string, fallback: number) => {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toggleStringSelection = (values: string[], value: string) => (
  values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value]
);

const getSearchResultKey = (result: SearchResult, index: number) => (
  result.chunk_id
  ?? `${result.doc_id ?? 'doc'}-${result.filepath ?? 'path'}-${index}`
);

const normalizeSearchResults = (
  contextItems?: Array<Partial<SearchResult>>,
  contexts?: string[],
): SearchResult[] => {
  if (Array.isArray(contextItems) && contextItems.length > 0) {
    return contextItems.map((item) => ({
      text: item.text ?? '',
      score: typeof item.score === 'number' ? item.score : 0,
      filepath: item.filepath,
      doc_id: item.doc_id,
      chunk_id: item.chunk_id,
    }));
  }

  if (Array.isArray(contexts) && contexts.length > 0) {
    return contexts.map((text) => ({ text, score: 0 }));
  }

  return [];
};

export default function RetrievalPage() {
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [collectionName, setCollectionName] = useState('');
  const [embedModelPath, setEmbedModelPath] = useState('');
  const [embedDim, setEmbedDim] = useState(DEFAULT_EMBED_DIM);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [filepath, setFilepath] = useState<string[]>([]);
  const [docId, setDocId] = useState<string[]>([]);
  const [useHybridSearch, setUseHybridSearch] = useState(true);

  // Filter options aligned with Chat page (from collection metadata)
  const [availableFilepaths, setAvailableFilepaths] = useState<string[]>([]);
  const [availableDocIds, setAvailableDocIds] = useState<string[]>([]);
  const [loadingCollectionData, setLoadingCollectionData] = useState(false);

  // Rerank params
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const [rerankType, setRerankType] = useState<'cross_encoder'>('cross_encoder');
  const [rerankModelPath, setRerankModelPath] = useState('');
  const [rerankDevice, setRerankDevice] = useState('cpu');
  const [rerankCandidateK, setRerankCandidateK] = useState(DEFAULT_RERANK_CANDIDATE_K);
  const [rerankTopK, setRerankTopK] = useState(DEFAULT_RERANK_TOP_K);

  // RAG params
  const [llmApiBase, setLlmApiBase] = useState('');
  const [llmModelName, setLlmModelName] = useState('');
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_MAX_NEW_TOKENS);

  // Results
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [ragAnswer, setRagAnswer] = useState('');
  const [ragContexts, setRagContexts] = useState<string[]>([]);
  const [ragContextItems, setRagContextItems] = useState<SearchResult[]>([]);

  // File generation
  const [inputPath, setInputPath] = useState('');
  const [outputPath, setOutputPath] = useState('');
  const [fileResult, setFileResult] = useState<any>(null);

  const selectedFilepathSet = useMemo(() => new Set(filepath), [filepath]);
  const selectedDocIdSet = useMemo(() => new Set(docId), [docId]);

  const clearFilters = useCallback(() => {
    setFilepath([]);
    setDocId([]);
  }, []);

  const toggleFilepath = useCallback((value: string) => {
    setFilepath((current) => toggleStringSelection(current, value));
  }, []);

  const toggleDocId = useCallback((value: string) => {
    setDocId((current) => toggleStringSelection(current, value));
  }, []);

  const tabFill = (
    setter: (value: string) => void,
  ) => (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key !== 'Tab') return;
    const current = e.currentTarget;
    if (!current.value?.trim() && current.placeholder?.trim()) {
      setter(current.placeholder);
    }
  };

  useEffect(() => {
    const fetchCollectionData = async () => {
      const normalizedCollectionName = collectionName.trim();
      if (!normalizedCollectionName) {
        setAvailableFilepaths([]);
        setAvailableDocIds([]);
        return;
      }
      setLoadingCollectionData(true);
      try {
        const response = await api.listCollections();
        if (response.success) {
          const collection = response.data.collections.find((c: any) => c.name === normalizedCollectionName);
          if (collection) {
            setAvailableFilepaths(collection.filepaths || []);
            setAvailableDocIds(collection.doc_ids || []);
          } else {
            setAvailableFilepaths([]);
            setAvailableDocIds([]);
          }
        }
      } finally {
        setLoadingCollectionData(false);
      }
    };

    const timeoutId = setTimeout(fetchCollectionData, 300);
    return () => clearTimeout(timeoutId);
  }, [collectionName]);

  useEffect(() => {
    setFilepath((current) => current.filter((item) => availableFilepaths.includes(item)));
    setDocId((current) => current.filter((item) => availableDocIds.includes(item)));
  }, [availableDocIds, availableFilepaths]);

  const handleSearch = async () => {
    const normalizedQuery = query.trim();
    const normalizedCollectionName = collectionName.trim();
    if (!normalizedQuery || !normalizedCollectionName) {
      toast.error('Please provide query and collection name');
      return;
    }

    setLoading(true);
    setSearchResults([]);
    try {
      const data: any = {
        query: normalizedQuery,
        collection_name: normalizedCollectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
        rerank_enabled: rerankEnabled,
        rerank_type: 'cross_encoder',
        rerank_model_path: rerankEnabled ? (rerankModelPath || undefined) : undefined,
        rerank_device: rerankDevice,
        rerank_candidate_k: rerankEnabled ? rerankCandidateK : undefined,
        rerank_top_k: rerankEnabled ? rerankTopK : undefined,
      };

      if (filepath.length > 0) data.filepath = filepath;
      if (docId.length > 0) data.doc_id = docId;

      const response = await api.search(data);
      if (response.success) {
        setSearchResults(response.data.results || []);
        toast.success(`Found ${response.data.results?.length || 0} results`);
      } else {
        toast.error('Search failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    const normalizedQuery = query.trim();
    const normalizedCollectionName = collectionName.trim();
    if (!normalizedQuery || !normalizedCollectionName) {
      toast.error('Please provide query and collection name');
      return;
    }

    setLoading(true);
    setRagAnswer('');
    setRagContexts([]);
    setRagContextItems([]);
    try {
      const data: any = {
        query: normalizedQuery,
        collection_name: normalizedCollectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
        rerank_enabled: rerankEnabled,
        rerank_type: 'cross_encoder',
        rerank_model_path: rerankEnabled ? (rerankModelPath || undefined) : undefined,
        rerank_device: rerankDevice,
        rerank_candidate_k: rerankEnabled ? rerankCandidateK : undefined,
        rerank_top_k: rerankEnabled ? rerankTopK : undefined,
        llm_api_base: llmApiBase || undefined,
        llm_model_name: llmModelName || undefined,
        temperature,
        max_new_tokens: maxNewTokens,
      };

      if (filepath.length > 0) data.filepath = filepath;
      if (docId.length > 0) data.doc_id = docId;

      const response = await api.generate(data);
      if (response.success) {
        const nextContexts = response.data.contexts || [];
        const nextContextItems = normalizeSearchResults(response.data.context_items, nextContexts);

        setRagAnswer(response.data.answer || '');
        setRagContexts(nextContexts);
        setRagContextItems(nextContextItems);
        setSearchResults(nextContextItems);
        toast.success('Generation completed');
      } else {
        toast.error('Generation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateFile = async () => {
    const normalizedInputPath = inputPath.trim();
    const normalizedOutputPath = outputPath.trim();
    const normalizedCollectionName = collectionName.trim();
    if (!normalizedInputPath || !normalizedOutputPath || !normalizedCollectionName) {
      toast.error('Please fill all required fields');
      return;
    }

    setLoading(true);
    try {
      const data: any = {
        input_path: normalizedInputPath,
        output_path: normalizedOutputPath,
        collection_name: normalizedCollectionName,
        embed_model_path: embedModelPath || undefined,
        embed_dim: embedDim,
        top_k: topK,
        use_hybrid_search: useHybridSearch,
        rerank_enabled: rerankEnabled,
        rerank_type: 'cross_encoder',
        rerank_model_path: rerankEnabled ? (rerankModelPath || undefined) : undefined,
        rerank_device: rerankDevice,
        rerank_candidate_k: rerankEnabled ? rerankCandidateK : undefined,
        rerank_top_k: rerankEnabled ? rerankTopK : undefined,
        llm_api_base: llmApiBase || undefined,
        llm_model_name: llmModelName || undefined,
        temperature,
        max_new_tokens: maxNewTokens,
      };

      const response = await api.generateFile(data);
      if (response.success) {
        setFileResult(response.data);
        toast.success(`Batch completed! Processed ${response.data.total_processed} items`);
      } else {
        toast.error('Batch run failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6 pb-12">
      <div className="mb-8 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Retrieval & RAG</h1>
          <p className="text-slate-600">Vector retrieval and retrieval-augmented generation (aligned with retrieval_api.py)</p>
        </div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" size="icon" title="Retrieval Settings">
              <Settings2 className="w-4 h-4" />
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl sm:max-w-2xl">
            <DialogHeader>
              <DialogTitle>Retrieval Model Settings</DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label>Embedding model path</Label>
                <Input value={embedModelPath} onChange={(e) => setEmbedModelPath(e.target.value)} onKeyDown={tabFill(setEmbedModelPath)} placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5" />
              </div>
              <div>
                <Label>Embed Dim</Label>
                        <Input type="number" value={embedDim} onChange={(e) => setEmbedDim(parseIntegerInput(e.target.value, DEFAULT_EMBED_DIM))} />
              </div>
              <div>
                <Label>LLM API Base</Label>
                <Input value={llmApiBase} onChange={(e) => setLlmApiBase(e.target.value)} onKeyDown={tabFill(setLlmApiBase)} placeholder="http://localhost:8005/v1" />
              </div>
              <div>
                <Label>LLM Model</Label>
                <Input value={llmModelName} onChange={(e) => setLlmModelName(e.target.value)} onKeyDown={tabFill(setLlmModelName)} placeholder="Qwen2.5-7B-Instruct" />
              </div>
              <div>
                <Label>CrossEncoder model path</Label>
                <Input value={rerankModelPath} onChange={(e) => setRerankModelPath(e.target.value)} onKeyDown={tabFill(setRerankModelPath)} placeholder="/data/h50056789/Rag_Chunking/model/BAAI/bge-reranker-v2-m3" />
              </div>
              <div>
                <Label>Rerank Device</Label>
                <Input value={rerankDevice} onChange={(e) => setRerankDevice(e.target.value)} placeholder="cuda:1" />
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Tabs defaultValue="single">
        <TabsList className="mb-6">
          <TabsTrigger value="single">Single Run</TabsTrigger>
          <TabsTrigger value="batch">Batch Run</TabsTrigger>
        </TabsList>

        {/* Single Query Tab */}
        <TabsContent value="single">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: Configuration */}
            <Card className="p-6 lg:col-span-1">
              <h2 className="font-bold mb-4">Configuration</h2>
              <ScrollArea className="h-[600px] pr-4">
                <div className="space-y-4">
                  <div>
                    <Label>Query</Label>
                    <Textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={tabFill(setQuery)}
                      placeholder="Type query..."
                      className="min-h-[100px]"
                    />
                  </div>

                  <div>
                    <Label>Collection Name</Label>
                    <Input
                      value={collectionName}
                      onChange={(e) => setCollectionName(e.target.value)}
                      onKeyDown={tabFill(setCollectionName)}
                      placeholder="test"
                    />
                  </div>

                  <div>
                    <Label>Embedding model path (optional)</Label>
                    <Input
                      value={embedModelPath}
                      onChange={(e) => setEmbedModelPath(e.target.value)}
                      onKeyDown={tabFill(setEmbedModelPath)}
                      placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Embed Dim</Label>
                      <Input
                        type="number"
                        value={embedDim}
                        onChange={(e) => setEmbedDim(parseIntegerInput(e.target.value, DEFAULT_EMBED_DIM))}
                      />
                    </div>
                    <div>
                      <Label>Top K</Label>
                      <Input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(parseIntegerInput(e.target.value, DEFAULT_TOP_K))}
                      />
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                        <Filter className="w-3 h-3" />
                        Metadata Filters (aligned with Chat)
                      </div>
                      {(filepath.length > 0 || docId.length > 0) && (
                        <Button variant="ghost" size="sm" onClick={clearFilters} className="h-6 px-2 text-xs">
                          <X className="w-3 h-3 mr-1" />Clear
                        </Button>
                      )}
                    </div>

                    <div>
                      <Label className="text-xs">Filepath</Label>
                      {availableFilepaths.length > 0 ? (
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline" className="w-full justify-between mt-1 h-auto min-h-9 text-xs" disabled={loadingCollectionData}>
                              <span className="truncate">{filepath.length === 0 ? 'Select filepath(s)...' : `${filepath.length} selected`}</span>
                              <ChevronsUpDown className="ml-2 h-3 w-3 shrink-0 opacity-50" />
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-80 p-0" align="start">
                            <Command>
                              <CommandInput placeholder="Search filepath..." className="h-8 text-xs" />
                              <CommandList>
                                <CommandEmpty>No filepath found.</CommandEmpty>
                                <CommandGroup>
                                  {availableFilepaths.map((fp) => (
                                    <CommandItem
                                      key={fp}
                                      onSelect={() => toggleFilepath(fp)}
                                      className="text-xs"
                                    >
                                      <Checkbox checked={selectedFilepathSet.has(fp)} className="mr-2" />
                                      <span className="truncate font-mono">{fp}</span>
                                    </CommandItem>
                                  ))}
                                </CommandGroup>
                              </CommandList>
                            </Command>
                          </PopoverContent>
                        </Popover>
                      ) : (
                        <div className="text-xs text-slate-400 mt-1 p-2 border rounded bg-slate-50">
                          {loadingCollectionData ? 'Loading collection metadata...' : 'No filepath metadata available'}
                        </div>
                      )}
                    </div>

                    <div>
                      <Label className="text-xs">Doc ID</Label>
                      {availableDocIds.length > 0 ? (
                        <Popover>
                          <PopoverTrigger asChild>
                            <Button variant="outline" className="w-full justify-between mt-1 h-auto min-h-9 text-xs" disabled={loadingCollectionData}>
                              <span className="truncate">{docId.length === 0 ? 'Select doc_id(s)...' : `${docId.length} selected`}</span>
                              <ChevronsUpDown className="ml-2 h-3 w-3 shrink-0 opacity-50" />
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-80 p-0" align="start">
                            <Command>
                              <CommandInput placeholder="Search doc_id..." className="h-8 text-xs" />
                              <CommandList>
                                <CommandEmpty>No doc_id found.</CommandEmpty>
                                <CommandGroup>
                                  {availableDocIds.map((id) => (
                                    <CommandItem
                                      key={id}
                                      onSelect={() => toggleDocId(id)}
                                      className="text-xs"
                                    >
                                      <Checkbox checked={selectedDocIdSet.has(id)} className="mr-2" />
                                      <span className="truncate font-mono">{id}</span>
                                    </CommandItem>
                                  ))}
                                </CommandGroup>
                              </CommandList>
                            </Command>
                          </PopoverContent>
                        </Popover>
                      ) : (
                        <div className="text-xs text-slate-400 mt-1 p-2 border rounded bg-slate-50">
                          {loadingCollectionData ? 'Loading collection metadata...' : 'No doc_id metadata available'}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="space-y-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-sm">RRF Rerank (Hybrid Search)</Label>
                        <p className="text-xs text-slate-500">Dense + Sparse + RRF (server ranker)</p>
                      </div>
                      <Switch checked={useHybridSearch} onCheckedChange={setUseHybridSearch} />
                    </div>

                      <div className="border-t pt-3">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <Label className="text-sm">CrossEncoder Rerank</Label>
                          <p className="text-xs text-slate-500">Semantic rerank for retrieval candidates</p>
                        </div>
                        <Switch checked={rerankEnabled} onCheckedChange={setRerankEnabled} />
                      </div>

                      {rerankEnabled && (
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <Label className="text-xs">Rerank Type</Label>
                            <Select value={rerankType} onValueChange={(v: 'cross_encoder') => setRerankType(v)}>
                              <SelectTrigger><SelectValue /></SelectTrigger>
                              <SelectContent>
                                <SelectItem value="cross_encoder">CrossEncoder</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <Label className="text-xs">Rerank Top K</Label>
                          <Input type="number" value={rerankTopK} onChange={(e) => setRerankTopK(parseIntegerInput(e.target.value, DEFAULT_RERANK_TOP_K))} />
                          </div>
                          <div className="col-span-2">
                            <Label className="text-xs">Rerank Candidate K</Label>
                          <Input type="number" value={rerankCandidateK} onChange={(e) => setRerankCandidateK(parseIntegerInput(e.target.value, DEFAULT_RERANK_CANDIDATE_K))} />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="border-t pt-4 mt-4">
                    <h3 className="font-medium mb-3 text-sm">LLM Configuration (required for RAG)</h3>
                    <div className="space-y-3">
                      <div>
                        <Label className="text-sm">API Base (optional)</Label>
                        <Input
                          value={llmApiBase}
                          onChange={(e) => setLlmApiBase(e.target.value)}
                          onKeyDown={tabFill(setLlmApiBase)}
                          placeholder="http://localhost:8005/v1"
                        />
                      </div>
                      <div>
                        <Label className="text-sm">Model Name (optional)</Label>
                        <Input
                          value={llmModelName}
                          onChange={(e) => setLlmModelName(e.target.value)}
                          onKeyDown={tabFill(setLlmModelName)}
                          placeholder="Qwen2.5-7B-Instruct"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <Label className="text-sm">Temperature</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloatInput(e.target.value, DEFAULT_TEMPERATURE))}
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Max Tokens</Label>
                          <Input
                            type="number"
                            value={maxNewTokens}
                            onChange={(e) => setMaxNewTokens(parseIntegerInput(e.target.value, DEFAULT_MAX_NEW_TOKENS))}
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
                      Search Only
                    </Button>
                    <Button
                      onClick={handleGenerate}
                      disabled={loading}
                      className="bg-gradient-to-r from-purple-600 to-pink-600"
                    >
                      <Sparkles className="w-4 h-4 mr-2" />
                      Search + Generate
                    </Button>
                  </div>
                </div>
              </ScrollArea>
            </Card>

            {/* Right: Results */}
            <div className="lg:col-span-2 space-y-6">
              {/* Search Results */}
              <Card className="p-6 h-[692px] flex flex-col overflow-hidden">
                <h2 className="font-bold mb-4 flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Search Results
                  {useHybridSearch && (
                    <Badge className="bg-blue-600">Hybrid Search</Badge>
                  )}
                  {rerankEnabled && (
                    <Badge className="bg-emerald-600">CrossEncoder</Badge>
                  )}
                </h2>
                {loading ? (
                  <div className="flex-1 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : searchResults.length > 0 ? (
                  <ScrollArea className="flex-1 min-h-0 pr-4">
                    <div className="space-y-3">
                      {searchResults.map((result, index) => (
                        <div
                          key={getSearchResultKey(result, index)}
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
                          <p className="text-sm text-slate-700 break-words">{result.text}</p>
                          {result.doc_id && (
                            <div className="text-xs text-slate-500 mt-2">
                              Doc ID: {result.doc_id}
                            </div>
                          )}
                          {result.chunk_id && (
                            <div className="text-xs text-slate-500 mt-1">
                              Chunk ID: {result.chunk_id}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-slate-400">
                    <div className="text-center">
                      <Search className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Search results will appear here</p>
                    </div>
                  </div>
                )}
              </Card>

              {/* RAG Answer */}
              {ragAnswer && (
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-purple-600" />
                    RAG Answer
                  </h2>
                  <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200 mb-4">
                    <p className="text-slate-800 whitespace-pre-wrap break-words">{ragAnswer}</p>
                  </div>

                  <Collapsible>
                    <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900">
                      <ChevronDown className="w-4 h-4" />
                      Show Used Contexts ({ragContexts.length} items)
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-4">
                      <div className="space-y-2">
                        {ragContextItems.map((item, index) => (
                          <div
                            key={getSearchResultKey(item, index)}
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
                            <p className="text-slate-700 break-words">{item.text}</p>
                            {(item.doc_id || item.chunk_id) && (
                              <div className="mt-2 space-y-1 text-xs text-slate-500">
                                {item.doc_id && <div>Doc ID: {item.doc_id}</div>}
                                {item.chunk_id && <div>Chunk ID: {item.chunk_id}</div>}
                              </div>
                            )}
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
          <Card className="p-6">
            <h2 className="font-bold mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Batch RAG Generation
            </h2>
            <div className="space-y-4 mb-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label>Input file path (.jsonl)</Label>
                  <div className="flex gap-2">
                    <Input
                      value={inputPath}
                      onChange={(e) => setInputPath(e.target.value)}
                      onKeyDown={tabFill(setInputPath)}
                      placeholder="/path/to/input.jsonl"
                    />
                    <PathPickerButton
                      mode="file"
                      value={inputPath}
                      title="Select Batch Input File"
                      description="This browser reads the filesystem on the machine running the backend service."
                      onSelect={setInputPath}
                    />
                  </div>
                </div>
                <div>
                  <Label>Output file path (.json)</Label>
                  <div className="flex gap-2">
                    <Input
                      value={outputPath}
                      onChange={(e) => setOutputPath(e.target.value)}
                      onKeyDown={tabFill(setOutputPath)}
                      placeholder="/path/to/output.json"
                    />
                    <PathPickerButton
                      mode="directory"
                      value={outputPath}
                      title="Select Output Directory"
                      description="Pick the folder to write the output file into, then edit the filename if needed."
                      onSelect={(selectedDirectory) => {
                        const currentName = outputPath.split(/[\\/]/).filter(Boolean).at(-1);
                        const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                        setOutputPath(
                          currentName && currentName.includes('.')
                            ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                            : selectedDirectory,
                        );
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label>Collection Name</Label>
                  <Input
                    value={collectionName}
                    onChange={(e) => setCollectionName(e.target.value)}
                    onKeyDown={tabFill(setCollectionName)}
                    placeholder="test"
                  />
                </div>
                <div>
                  <Label>Top K</Label>
                  <Input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseIntegerInput(e.target.value, DEFAULT_TOP_K))}
                  />
                </div>
                <div>
                  <Label>Embed Dim</Label>
                  <Input
                    type="number"
                    value={embedDim}
                    onChange={(e) => setEmbedDim(parseIntegerInput(e.target.value, DEFAULT_EMBED_DIM))}
                  />
                </div>
              </div>

              <div>
                <Label>Embedding model path (optional)</Label>
                <Input
                  value={embedModelPath}
                  onChange={(e) => setEmbedModelPath(e.target.value)}
                  onKeyDown={tabFill(setEmbedModelPath)}
                  placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
                />
              </div>

              <div className="space-y-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-center justify-between">
                  <div>
                    <Label className="text-sm">RRF Rerank (Hybrid Search)</Label>
                    <p className="text-xs text-slate-500">Dense + Sparse + RRF</p>
                  </div>
                  <Switch checked={useHybridSearch} onCheckedChange={setUseHybridSearch} />
                </div>

                <div className="border-t pt-3">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <Label className="text-sm">CrossEncoder Rerank</Label>
                      <p className="text-xs text-slate-500">Semantic rerank for retrieval candidates</p>
                    </div>
                    <Switch checked={rerankEnabled} onCheckedChange={setRerankEnabled} />
                  </div>

                  {rerankEnabled && (
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <Label className="text-xs">Rerank Type</Label>
                        <Select value={rerankType} onValueChange={(v: 'cross_encoder') => setRerankType(v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="cross_encoder">CrossEncoder</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-xs">Rerank Device</Label>
                        <Input value={rerankDevice} onChange={(e) => setRerankDevice(e.target.value)} onKeyDown={tabFill(setRerankDevice)} placeholder="cuda:1" />
                      </div>
                      <div>
                        <Label className="text-xs">Rerank Top K</Label>
                        <Input type="number" value={rerankTopK} onChange={(e) => setRerankTopK(parseIntegerInput(e.target.value, DEFAULT_RERANK_TOP_K))} />
                      </div>
                      <div>
                        <Label className="text-xs">Rerank Candidate K</Label>
                        <Input type="number" value={rerankCandidateK} onChange={(e) => setRerankCandidateK(parseIntegerInput(e.target.value, DEFAULT_RERANK_CANDIDATE_K))} />
                      </div>
                      <div className="col-span-2">
                        <Label className="text-xs">CrossEncoder model path (optional)</Label>
                        <Input value={rerankModelPath} onChange={(e) => setRerankModelPath(e.target.value)} onKeyDown={tabFill(setRerankModelPath)} placeholder="/data/h50056789/Rag_Chunking/model/BAAI/bge-reranker-v2-m3" />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-t pt-4">
                <div>
                  <Label>LLM API Base (optional)</Label>
                  <Input value={llmApiBase} onChange={(e) => setLlmApiBase(e.target.value)} onKeyDown={tabFill(setLlmApiBase)} placeholder="http://localhost:8005/v1" />
                </div>
                <div>
                  <Label>LLM Model Name (optional)</Label>
                  <Input value={llmModelName} onChange={(e) => setLlmModelName(e.target.value)} onKeyDown={tabFill(setLlmModelName)} placeholder="Qwen2.5-7B-Instruct" />
                </div>
                <div>
                  <Label>Temperature</Label>
                  <Input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloatInput(e.target.value, DEFAULT_TEMPERATURE))} />
                </div>
                <div>
                  <Label>Max New Tokens</Label>
                  <Input type="number" value={maxNewTokens} onChange={(e) => setMaxNewTokens(parseIntegerInput(e.target.value, DEFAULT_MAX_NEW_TOKENS))} />
                </div>
              </div>

              <div className="p-3 rounded-lg border bg-slate-50 text-xs text-slate-600">
                Pipeline mapping:{' '}
                <span className="font-medium">chunk-file output {'->'} index build input</span>
                {' '}and{' '}
                <span className="font-medium">retrieval output {'->'} retrieval eval</span>.
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
                  Running batch...
                </>
              ) : (
                'Start Batch Generation'
              )}
            </Button>

            {fileResult && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-medium mb-2 text-green-900">Completed</h3>
                <div className="text-sm space-y-1 text-green-800">
                  <p><strong>Output file:</strong> {fileResult.output_file}</p>
                  <p><strong>Processed:</strong> {fileResult.total_processed}</p>
                  <p><strong>Failed:</strong> {fileResult.total_failed}</p>
                  {fileResult.message && <p><strong>Message:</strong> {fileResult.message}</p>}
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

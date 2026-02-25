import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Send, 
  Loader2, 
  Sparkles, 
  Settings2, 
  Database,
  Zap,
  FileText,
  ChevronDown,
  Copy,
  Check,
  Filter,
  X
} from 'lucide-react';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import { Switch } from '../components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '../components/ui/collapsible';
import { api } from '../utils/api';
import { toast } from 'sonner';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  contexts?: Array<{
    text: string;
    score: number;
    filepath?: string;
    doc_id?: string;
  }>;
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Settings
  const [collectionName, setCollectionName] = useState('');
  const [embedModelPath, setEmbedModelPath] = useState('');
  const [embedDim, setEmbedDim] = useState(1024);
  const [topK, setTopK] = useState(5);
  const [enableRag, setEnableRag] = useState(true);
  const [useHybridSearch, setUseHybridSearch] = useState(false);
  const [llmApiBase, setLlmApiBase] = useState('http://localhost:8005/v1');
  const [llmModelName, setLlmModelName] = useState('');
  const [temperature, setTemperature] = useState(0.1);
  const [maxNewTokens, setMaxNewTokens] = useState(1280);

  // Document Filters
  const [filterFilepath, setFilterFilepath] = useState('');
  const [filterDocId, setFilterDocId] = useState('');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const clearFilters = () => {
    setFilterFilepath('');
    setFilterDocId('');
  };

  const handleSend = async () => {
    // Validate required fields
    if (!input.trim()) {
      toast.error('Please enter a question');
      return;
    }
    
    // LLM settings are always required
    if (!llmModelName) {
      toast.error('Please configure LLM Model Name in settings first');
      setShowSettings(true);
      return;
    }
    
    // Collection and embed model are required only when RAG is enabled
    if (enableRag) {
      if (!collectionName) {
        toast.error('Please configure Collection Name in settings (required when RAG is enabled)');
        setShowSettings(true);
        return;
      }
      if (!embedModelPath) {
        toast.error('Please configure Embedding Model Path in settings (required when RAG is enabled)');
        setShowSettings(true);
        return;
      }
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const data: any = {
        query: input,
        collection_name: collectionName,
        embed_model_path: embedModelPath,
        embed_dim: embedDim,
        top_k: topK,
        enable_rag: enableRag,
        use_hybrid_search: useHybridSearch || undefined,
        filepath: filterFilepath || undefined,
        doc_id: filterDocId || undefined,
        llm_api_base: llmApiBase,
        llm_model_name: llmModelName,
        temperature,
        max_new_tokens: maxNewTokens,
      };

      const response = await api.generate(data);
      
      if (response.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: response.data.answer || 'Sorry, I couldn\'t generate an answer.',
          contexts: response.data.context_items || [],
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        toast.error('Generation failed: ' + response.message);
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: `Sorry, an error occurred: ${response.message}`,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      toast.error('Request failed');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="flex h-full">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="border-b bg-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg">RAG Assistant</h1>
              <p className="text-xs text-slate-500 flex items-center gap-1">
                {enableRag ? (
                  <>
                    {collectionName ? `Collection: ${collectionName}` : 'Configure collection in settings'}
                    {useHybridSearch && <Badge className="ml-1 h-4 text-xs">Hybrid</Badge>}
                  </>
                ) : (
                  <>
                    Pure LLM Mode
                    <Badge variant="secondary" className="ml-1 h-4 text-xs">RAG Disabled</Badge>
                  </>
                )}
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
          >
            <Settings2 className="w-4 h-4 mr-2" />
            {showSettings ? 'Hide' : 'Show'} Settings
          </Button>
        </header>

        {/* Messages */}
        <div className="flex-1 px-6 py-8 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <Sparkles className="w-10 h-10 text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold mb-3">Start Conversation</h2>
                <p className="text-slate-500 mb-6">
                  Ask questions and get intelligent answers from your knowledge base
                </p>
                <div className="grid grid-cols-1 gap-3 text-left">
                  <button
                    onClick={() => setInput('What is a vector database?')}
                    className="p-3 bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:shadow-md transition-all text-sm"
                  >
                    What is a vector database?
                  </button>
                  <button
                    onClick={() => setInput('How does RAG work?')}
                    className="p-3 bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:shadow-md transition-all text-sm"
                  >
                    How does RAG work?
                  </button>
                  <button
                    onClick={() => setInput('How to optimize retrieval quality?')}
                    className="p-3 bg-white border border-slate-200 rounded-lg hover:border-blue-300 hover:shadow-md transition-all text-sm"
                  >
                    How to optimize retrieval quality?
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-6">
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-3xl ${message.type === 'user' ? 'w-auto' : 'w-full'}`}>
                      {message.type === 'user' ? (
                        <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3">
                          <p className="whitespace-pre-wrap">{message.content}</p>
                        </div>
                      ) : (
                        <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm p-4 shadow-sm">
                          <div className="flex items-start gap-3 mb-3">
                            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center flex-shrink-0">
                              <Sparkles className="w-4 h-4 text-white" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-slate-800 whitespace-pre-wrap">{message.content}</p>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => copyToClipboard(message.content, message.id)}
                              className="flex-shrink-0"
                            >
                              {copiedId === message.id ? (
                                <Check className="w-4 h-4 text-green-600" />
                              ) : (
                                <Copy className="w-4 h-4" />
                              )}
                            </Button>
                          </div>

                          {message.contexts && message.contexts.length > 0 && (
                            <Collapsible>
                              <CollapsibleTrigger className="flex items-center gap-2 text-xs font-medium text-slate-500 hover:text-slate-700 mt-3 pt-3 border-t">
                                <FileText className="w-3 h-3" />
                                View Sources ({message.contexts.length})
                                <ChevronDown className="w-3 h-3" />
                              </CollapsibleTrigger>
                              <CollapsibleContent className="mt-3 space-y-2">
                                {message.contexts.map((ctx, idx) => (
                                  <div
                                    key={idx}
                                    className="p-3 bg-slate-50 rounded-lg text-xs border border-slate-100"
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <Badge variant="outline" className="text-xs">
                                        Source {idx + 1}
                                      </Badge>
                                      <span className="text-blue-600 font-medium">
                                        Score: {ctx.score.toFixed(4)}
                                      </span>
                                    </div>
                                    <p className="text-slate-700">{ctx.text}</p>
                                    {ctx.filepath && (
                                      <p className="text-slate-400 mt-1">📄 {ctx.filepath}</p>
                                    )}
                                  </div>
                                ))}
                              </CollapsibleContent>
                            </Collapsible>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm p-4 shadow-sm">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                        <Loader2 className="w-4 h-4 text-white animate-spin" />
                      </div>
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t bg-white px-6 py-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything..."
                  className="pr-12 h-12 text-base"
                  disabled={loading}
                />
              </div>
              <Button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="h-12 px-6 bg-gradient-to-r from-blue-600 to-purple-600"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <aside className="w-80 border-l bg-white flex flex-col">
          <div className="p-4 border-b">
            <h2 className="font-bold flex items-center gap-2">
              <Settings2 className="w-5 h-5" />
              Configuration
            </h2>
          </div>
          
          <div className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-6">
              {/* RAG Toggle */}
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
                <div>
                  <Label className="text-sm font-medium">Enable RAG</Label>
                  <p className="text-xs text-slate-500">Vector Retrieval + Generation</p>
                </div>
                <Switch
                  checked={enableRag}
                  onCheckedChange={setEnableRag}
                />
              </div>

              {/* Index Settings - Only show when RAG is enabled */}
              {enableRag && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <Database className="w-4 h-4" />
                    Index Settings
                  </div>
                  <div>
                    <Label className="text-xs">
                      Collection Name <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      value={collectionName}
                      onChange={(e) => setCollectionName(e.target.value)}
                      placeholder="my_collection"
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">
                      Embedding Model Path <span className="text-red-500">*</span>
                    </Label>
                    <Input
                      value={embedModelPath}
                      onChange={(e) => setEmbedModelPath(e.target.value)}
                      placeholder="/path/to/bge-large-en-v1.5"
                      className="mt-1"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs">Embed Dim</Label>
                      <Input
                        type="number"
                        value={embedDim}
                        onChange={(e) => setEmbedDim(parseInt(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Top K</Label>
                      <Input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(parseInt(e.target.value))}
                        className="mt-1"
                      />
                    </div>
                  </div>

                  {/* Document Filters */}
                  <div className="space-y-3 pt-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                        <Filter className="w-3 h-3" />
                        Filter Documents
                      </div>
                      {(filterFilepath || filterDocId) && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={clearFilters}
                          className="h-5 px-1 text-xs"
                        >
                          <X className="w-3 h-3" />
                        </Button>
                      )}
                    </div>

                    <div>
                      <Label className="text-xs">File Path (Priority)</Label>
                      <Input
                        value={filterFilepath}
                        onChange={(e) => {
                          setFilterFilepath(e.target.value);
                          if (e.target.value) setFilterDocId('');
                        }}
                        placeholder="e.g., /path/to/document.pdf"
                        className="mt-1 text-xs"
                      />
                    </div>

                    <div>
                      <Label className="text-xs">Document ID</Label>
                      <Input
                        value={filterDocId}
                        onChange={(e) => {
                          setFilterDocId(e.target.value);
                          if (e.target.value) setFilterFilepath('');
                        }}
                        placeholder="e.g., doc_123"
                        className="mt-1 text-xs"
                        disabled={!!filterFilepath}
                      />
                    </div>

                    {(filterFilepath || filterDocId) && (
                      <div className="p-2 bg-green-50 rounded border border-green-200">
                        <div className="text-xs font-medium text-green-900 mb-0.5">
                          Active Filter
                        </div>
                        <div className="text-xs text-green-700 truncate">
                          {filterFilepath ? `📄 ${filterFilepath}` : `🆔 ${filterDocId}`}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div>
                      <Label className="text-xs font-medium">Hybrid Search</Label>
                      <p className="text-xs text-slate-500">Dense + Sparse</p>
                    </div>
                    <Switch
                      checked={useHybridSearch}
                      onCheckedChange={setUseHybridSearch}
                    />
                  </div>
                </div>
              )}

              {/* LLM Settings - Always visible */}
              <div className="space-y-3 pt-3 border-t">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Zap className="w-4 h-4" />
                  LLM Settings
                </div>
                <div>
                  <Label className="text-xs">
                    API Base <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    value={llmApiBase}
                    onChange={(e) => setLlmApiBase(e.target.value)}
                    placeholder="http://localhost:8005/v1"
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label className="text-xs">
                    Model Name <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    value={llmModelName}
                    onChange={(e) => setLlmModelName(e.target.value)}
                    placeholder="/path/to/Qwen2.5-7B-Instruct"
                    className="mt-1"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs">Temperature</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Max Tokens</Label>
                    <Input
                      type="number"
                      value={maxNewTokens}
                      onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>

              {/* Info */}
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs text-slate-600">
                  💡 {enableRag 
                    ? 'Tip: Create a collection in the Index page first, then configure it here for RAG chatting.'
                    : 'Tip: RAG is disabled. Only LLM will be used without context retrieval.'}
                </p>
              </div>
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
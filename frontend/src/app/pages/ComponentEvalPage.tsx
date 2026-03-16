import React, { useState, useEffect, useMemo, useRef, useCallback, type KeyboardEvent } from 'react';
import { useLocation } from 'react-router';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Tabs, TabsContent } from '../components/ui/tabs';
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
import { Loader2, Cpu, Activity, FileText, Network, Grid3x3, Settings, Plus, X, BarChart3, Sparkles, Sigma, FlaskConical, ChevronDown, ChevronUp, Maximize2 } from 'lucide-react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { api } from '../utils/api';
import { toast } from 'sonner';
import ForceGraph2D from 'react-force-graph-2d';
import { PathPickerButton } from '../components/PathPickerButton';

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

const fillPlaceholderOnTab = (
  event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  currentValue: string,
  placeholderValue: string,
  setValue: (value: string) => void,
) => {
  if (event.key !== 'Tab' || currentValue.trim() || !placeholderValue) {
    return;
  }
  event.preventDefault();
  setValue(placeholderValue);
};

const parseMaxEvalChunksInput = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  if (!/^-?\d+$/.test(trimmed)) {
    throw new Error('Max eval chunks must be an integer. Use -1 for all chunks.');
  }
  const parsed = Number.parseInt(trimmed, 10);
  if (parsed === 0 || parsed < -1) {
    throw new Error('Max eval chunks must be -1 or a positive integer.');
  }
  return parsed;
};

const chunkJsonPlaceholder = `{
  "filepath": "./dataset/docs/2wikimqa/0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt",
  "splits": [
    [
      "Passage 1:\\nZoran Svonja ...\\n\\nPassage 2:\\nAnton Shunto ...",
      "djgashdkjghskdjghskjdghkjas",
      1
    ]
  ],
  "time_cost": 2.4059367179870605
}`;

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
              onKeyDown={(e) => fillPlaceholderOnTab(e, pplModelPath, e.currentTarget.placeholder, setPplModelPath)}
              placeholder="/models/Qwen2.5-7B-Instruct"
              className="mt-1.5"
            />
            <p className="text-xs text-slate-500 mt-1">Perplexity model for Boundary Clarity</p>
          </div>
          <div>
            <Label>Sim Model Path</Label>
            <Input
              value={simModelPath}
              onChange={(e) => setSimModelPath(e.target.value)}
              onKeyDown={(e) => fillPlaceholderOnTab(e, simModelPath, e.currentTarget.placeholder, setSimModelPath)}
              placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
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
                  onKeyDown={(e) => fillPlaceholderOnTab(e, vllmApiBase, e.currentTarget.placeholder, setVllmApiBase)}
                  placeholder="http://localhost:8005/v1"
                  className="mt-1.5"
                />
              </div>
              <div>
                <Label>vLLM Model Name</Label>
                <Input
                  value={vllmModelName}
                  onChange={(e) => setVllmModelName(e.target.value)}
                  onKeyDown={(e) => fillPlaceholderOnTab(e, vllmModelName, e.currentTarget.placeholder, setVllmModelName)}
                  placeholder="Qwen2.5-7B-Instruct"
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
              onKeyDown={(e) => fillPlaceholderOnTab(e, modelPath, e.currentTarget.placeholder, setModelPath)}
              placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
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
                  onKeyDown={(e) => fillPlaceholderOnTab(e, vllmApiBase, e.currentTarget.placeholder, setVllmApiBase)}
                  placeholder="http://localhost:8005/v1"
                  className="mt-1.5"
                />
              </div>
              <div>
                <Label>vLLM Model Name</Label>
                <Input
                  value={vllmModelName}
                  onChange={(e) => setVllmModelName(e.target.value)}
                  onKeyDown={(e) => fillPlaceholderOnTab(e, vllmModelName, e.currentTarget.placeholder, setVllmModelName)}
                  placeholder="Qwen2.5-7B-Instruct"
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
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const isRetrievalSection = location.search.includes('section=retrieval');
  const initialTab = isRetrievalSection ? 'retrieval' : 'quality';
  const [activeTab, setActiveTab] = useState<'quality' | 'stickiness' | 'retrieval'>(initialTab as 'quality' | 'stickiness' | 'retrieval');
  const [selectedChunkMetric, setSelectedChunkMetric] = useState<string | null>(null);

  type ChunkMetricDoc = {
    title: string;
    blurb: string;
    formulaLatex: string;
    interpretation: string | string[];
    variables?: string[];
    projectExample: string[];
    sources?: Array<{ label: string; url: string }>;
  };

  const chunkMetricSources = [
    { label: 'MoC repository', url: 'https://github.com/IAAR-Shanghai/Meta-Chunking' },
    { label: 'MoC paper (arXiv)', url: 'https://arxiv.org/abs/2503.09600' },
    { label: 'Structural entropy background', url: 'https://en.wikipedia.org/wiki/Entropy_(information_theory)' },
    { label: 'Cosine similarity background', url: 'https://en.wikipedia.org/wiki/Cosine_similarity' },
    { label: 'MoC metric script: chunk_eval.py', url: 'file:///F:/thesis/Meta-Chunking/MoC/our_metrics/chunk_eval.py' },
    { label: 'MoC metric script: relation_eval.py', url: 'file:///F:/thesis/Meta-Chunking/MoC/our_metrics/relation_eval.py' },
    { label: 'Project implementation: chunk_eval_refactored.py', url: 'file:///F:/thesis/Meta-Chunking/component_eval/chunk/chunk_eval_refactored.py' },
    { label: 'Project implementation: relation_eval_refactored.py', url: 'file:///F:/thesis/Meta-Chunking/component_eval/chunk/relation_eval_refactored.py' },
  ];

  const chunkMetricInfo: Record<string, ChunkMetricDoc> = {
    bc: {
      title: 'Boundary Clarity (BC)',
      blurb: 'Measures how clearly adjacent chunks are separated at the semantic level using conditional perplexity ratio.',
      formulaLatex: String.raw`BC_i=\frac{\mathrm{ppl}(c_{i+1}\mid c_i)}{\mathrm{ppl}(c_{i+1}\mid \varnothing)},\quad BC=\frac{1}{N-1}\sum_{i=1}^{N-1}BC_i`,
      interpretation:
        'Boundary Clarity evaluates the semantic independence between adjacent text chunks. If two chunks are strongly dependent, the previous chunk significantly reduces the perplexity of the next chunk, leading to a smaller BC value; if they are semantically independent, the conditional perplexity approaches the unconditional perplexity and BC approaches 1.',
      variables: [
        String.raw`c_i:\ \text{the }i\text{-th text chunk in the document}`,
        String.raw`c_{i+1}:\ \text{the next chunk following }c_i`,
        String.raw`\mathrm{ppl}(x):\ \text{perplexity of sequence }x\text{ computed by a language model}`,
        String.raw`\mathrm{ppl}(x\mid y):\ \text{conditional perplexity of }x\text{ given context }y`,
        String.raw`N:\ \text{total number of chunks in the document}`,
        String.raw`BC_i:\ \text{boundary clarity score for the boundary between }c_i\text{ and }c_{i+1}`,
        String.raw`BC:\ \text{average boundary clarity across all adjacent chunk pairs}`,
      ],
      projectExample: [
        'In this project, each adjacent chunk pair (c_i, c_{i+1}) produces one BC_i score.',
        'BC_i compares the perplexity of the next chunk with and without the previous chunk as context.',
        'The UI reports avg_boundary_clarity as the mean BC across all valid chunk boundaries.',
      ],
    },
    ds: {
      title: 'Semantic Dissimilarity (DS)',
      blurb: 'Quantifies the semantic difference between adjacent text chunks based on the cosine similarity of their embedding representations.',
      formulaLatex: String.raw`DS_i = 1 - \cos\left(\mathbf{e}(c_i),\mathbf{e}(c_{i+1})\right),\quad DS = \frac{1}{N-1}\sum_{i=1}^{N-1} DS_i`,
      interpretation:
        'DS reflects the degree of semantic separation between neighboring chunks. Higher DS indicates stronger semantic independence between adjacent chunks, suggesting clearer topical boundaries; extremely low DS values may imply over-segmentation where semantically similar content is unnecessarily split across chunks.',
      projectExample: [
        '`chunk_eval_refactored.py` computes sentence embeddings for each chunk and derives DS using 1 − cosine similarity between adjacent chunk embeddings.',
        'The evaluation dashboard aggregates DS across the dataset and reports the mean value as avg_semantic_dissimilarity.',
      ],
    },
    cs: {
      title: 'Chunk Stickiness (CS, Structural-Entropy View)',
      blurb: 'Measures the strength and structural coherence of semantic connections between chunks using structural entropy over a chunk relation graph.',
      formulaLatex: String.raw`CS=H(G_\tau)=-\sum_{v\in V}p(v)\log_2 p(v),\quad p(v)=\frac{\deg(v)}{\sum_{u\in V}\deg(u)}`,
      interpretation:
        'Chunk Stickiness evaluates how tightly text chunks are semantically connected across the entire document. Lower entropy indicates that connections concentrate around a few strongly related chunks, implying stronger semantic cohesion, while higher entropy suggests a more scattered connection pattern with weaker or less structured relationships between chunks.',
      variables: [
        String.raw`G_\tau:\ \text{thresholded semantic graph constructed from chunk relationships}`,
        String.raw`V:\ \text{set of nodes in the graph (each node is a chunk)}`,
        String.raw`v:\ \text{a node representing a single chunk}`,
        String.raw`\deg(v):\ \text{degree of node }v\text{ (number/weight of incident edges)}`,
        String.raw`p(v)=\frac{\deg(v)}{\sum_{u\in V}\deg(u)}:\ \text{normalized degree distribution}`,
        String.raw`H(G_\tau):\ \text{structural entropy of the chunk graph after threshold filtering}`,
        String.raw`\tau:\ \text{edge-weight threshold used to retain strong semantic connections}`,
      ],
      projectExample: [
        'In this project, chunks are first connected using pairwise semantic edge weights derived from perplexity-based similarity.',
        'Edges with weights greater than a predefined threshold are retained to construct the semantic graph.',
        '`relation_eval_refactored.py` computes node degrees and applies the structural entropy formula to the resulting graph.',
        'The UI reports two variants: structural_entropy_complete and structural_entropy_incomplete.',
      ],
    },
  };

  const retrievalMetricInfo: Record<string, ChunkMetricDoc> = {
    recip_rank: {
      title: 'MRR (Mean Reciprocal Rank)',
      blurb: 'Dataset-level mean of reciprocal rank; the backend field name is `recip_rank`, but the aggregated metric is MRR.',
      formulaLatex: String.raw`MRR=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}`,
      interpretation: 'Higher is better. For each query, the evaluator finds the first relevant retrieved item, computes its reciprocal rank, and then averages this value across all evaluated queries.',
      variables: [
        String.raw`Q:\ \text{query set}`,
        String.raw`rank_q:\ \text{rank of the first relevant hit for query }q`,
      ],
      projectExample: ['If first-hit ranks across queries are [1, 2, not-found], reciprocal ranks are [1, 1/2, 0], and MRR is their mean.'],
    },
    
    rprec: {
      title: 'R-Precision',
      blurb: 'Precision measured at rank R, where R equals the number of relevant items for the query.',
      formulaLatex: String.raw`R\text{-}Precision=\frac{|Rel\cap Ret_R|}{R}`,
      interpretation: 'Higher is better. Measures how many relevant chunks appear within the top-R retrieved results, where R is the total number of relevant chunks for the query.',
      variables: [
        String.raw`Rel:\ \text{set of relevant chunks from }\mathtt{gold\_reference}`,
        String.raw`Ret_R:\ \text{set of top-}R\text{ retrieved chunks}`,
        String.raw`R=|Rel|:\ \text{total number of relevant chunks for the query}`,
      ],
      projectExample: [
        'If a query has 3 relevant chunks (R = 3) and only 1 of them appears within the top-3 retrieved results, then R-Precision = 1/3.'
      ],
    },

    precision: {
      title: 'Precision@k',
      blurb: 'Proportion of retrieved items within the top-k results that are relevant.',
      formulaLatex: String.raw`Precision@k=\frac{|Rel\cap Ret_k|}{k}`,
      interpretation: 'Higher values indicate better ranking accuracy. Precision@k evaluates how many of the retrieved top-k items are relevant.',
      variables: [
        String.raw`Rel:\ \text{set of relevant chunks derived from }\mathtt{gold\_reference}`,
        String.raw`Ret_k:\ \text{set of top-}k\text{ retrieved chunks}`,
        String.raw`k:\ \text{retrieval cutoff rank}`
      ],
      projectExample: [
        'In this project, relevance is determined by matching (doc_id, chunk_id) between `rag_retrieval` and `gold_reference`.',
        'Precision@k is computed for each query at different cutoff values and then averaged across all queries.'
      ],
      sources: [
        { label: 'Wikipedia: Precision and recall', url: 'https://en.wikipedia.org/wiki/Precision_and_recall' },
        { label: 'Project implementation: eval_retrieval.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_retrieval.py' },
      ],
    },

    recall: {
      title: 'Recall@k',
      blurb: 'Proportion of all relevant items that are successfully retrieved within the top-k results.',
      formulaLatex: String.raw`Recall@k=\frac{|Rel\cap Ret_k|}{|Rel|}`,
      interpretation: 'Higher values indicate better coverage of relevant documents. Recall@k measures how many of the true relevant chunks are retrieved within the top-k results.',
      variables: [
        String.raw`Rel:\ \text{set of relevant chunks derived from }\mathtt{gold\_reference}`,
        String.raw`Ret_k:\ \text{set of top-}k\text{ retrieved chunks}`,
        String.raw`k:\ \text{retrieval cutoff rank}`
      ],
      projectExample: [
        'If a query has 3 gold chunks and the top-5 retrieval returns 2 of them, then Recall@5 = 2/3.',
        'Recall@k reflects retrieval coverage and is particularly important in RAG systems to ensure relevant context is included.'
      ],
      sources: [
        { label: 'Wikipedia: Precision and recall', url: 'https://en.wikipedia.org/wiki/Precision_and_recall' },
        { label: 'Project implementation: eval_retrieval.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_retrieval.py' },
      ],
    },

    map: {
      title: 'MAP@k (Mean Average Precision)',
      blurb: 'Mean of Average Precision across all queries, measuring how well relevant documents are ranked within the top-k retrieved results.',
      
      formulaLatex: String.raw`AP@k=\frac{1}{|Rel|}\sum_{i=1}^{k}P@i\cdot rel_i,\quad MAP@k=\frac{1}{|Q|}\sum_{q\in Q}AP_q@k`,
      
      interpretation: 'Higher values indicate better retrieval quality. MAP rewards systems that rank relevant documents early while also retrieving all relevant items within the top-k results.',
      
      variables: [
        String.raw`Q:\ \text{set of queries}`,
        String.raw`Rel:\ \text{number of relevant documents for a query}`,
        String.raw`P@i:\ \text{precision at rank } i`,
        String.raw`rel_i \in \{0,1\}:\ \text{relevance indicator at rank } i`,
      ],
      
      projectExample: [
        'For each query, AP@k accumulates the precision values at the ranks where relevant chunks appear among the top-k retrieval results. The MAP score is then obtained by averaging these AP values across all queries in the dataset.'
      ],
      
      sources: [
        { label: 'Introduction to Information Retrieval (Manning et al.)', url: 'https://nlp.stanford.edu/IR-book/' },
        { label: 'Project implementation: eval_retrieval.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_retrieval.py' },
      ],
    },
    mrr: {
      title: 'MRR@k (Mean Reciprocal Rank)',
      blurb: 'Measures how early the first relevant item appears in the ranked results.',
      formulaLatex: String.raw`MRR@k=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}`,
      interpretation: 'Higher is better. Strongly rewards systems that place at least one relevant chunk very early in the ranking.',
      variables: [
        String.raw`rank_q:\ \text{rank position of the first relevant item for query } q`,
        String.raw`Q:\ \text{set of evaluation queries}`
      ],
      projectExample: [
        'Example: if the first relevant results for three queries appear at ranks [1, 2, not-found],',
        'their reciprocal ranks are [1, 1/2, 0], so the MRR = (1 + 0.5 + 0) / 3 = 0.5.'
      ],
      sources: [
        { label: 'Wikipedia: Mean Reciprocal Rank', url: 'https://en.wikipedia.org/wiki/Mean_reciprocal_rank' },
        { label: 'Project implementation: eval_retrieval.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_retrieval.py' }
      ],
    },
    ndcg: {
      title: 'nDCG@k',
      blurb: 'Normalized Discounted Cumulative Gain measuring ranking quality with logarithmic position discounting.',
      formulaLatex: String.raw`DCG@k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},\quad nDCG@k=\frac{DCG@k}{IDCG@k}`,
      interpretation: 'Higher values indicate better ranking quality. nDCG rewards placing highly relevant items near the top of the ranked list while discounting lower-ranked positions.',
      variables: [
        String.raw`rel_i:\ \text{relevance score of the document at rank }i`,
        String.raw`k:\ \text{cutoff rank}`,
        String.raw`IDCG@k:\ \text{maximum possible DCG obtained by an ideal ranking up to }k`
      ],
      projectExample: [
        'Two retrieval systems may return the same number of relevant chunks, but the one ranking relevant chunks earlier will achieve a higher nDCG score.'
      ],
      sources: [
        { label: 'Wikipedia: Discounted cumulative gain', url: 'https://en.wikipedia.org/wiki/Discounted_cumulative_gain' },
        { label: 'Project implementation: eval_retrieval.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_retrieval.py' },
      ],
    },
  };

  // ── Chunk Quality – Direct Input ────────────────────────────────────────────
  const [qualityChunksJson, setQualityChunksJson] = useState('');
  const [enableSemanticSimilarity, setEnableSemanticSimilarity] = useState(true);
  const [enableBoundaryClarity, setEnableBoundaryClarity] = useState(true);
  const [qualityResult, setQualityResult] = useState<any>(null);

  // ── Chunk Quality – File Input ──────────────────────────────────────────────
  const [tempQualityPath, setTempQualityPath] = useState('');
  const [qualityOutputPath, setQualityOutputPath] = useState('');
  const [qualityMaxEvalChunks, setQualityMaxEvalChunks] = useState('-1');
  const [qualityFilePaths, setQualityFilePaths] = useState<string[]>([]);

  // ── Chunk Quality – Model Config (shared) ──────────────────────────────────
  const [qualityConfigOpen, setQualityConfigOpen] = useState(false);
  const [pplModelPath, setPplModelPath] = useState('');
  const [simModelPath, setSimModelPath] = useState('');
  const [qualityUseVllm, setQualityUseVllm] = useState(false);
  const [qualityVllmApiBase, setQualityVllmApiBase] = useState('http://localhost:8005/v1');
  const [qualityVllmModelName, setQualityVllmModelName] = useState('');

  // ── Chunk Stickiness – Direct Input ────────────────────────────────────────
  const [stickinessChunksJson, setStickinessChunksJson] = useState('');
  const [threshold, setThreshold] = useState([0.7]);
  const [delta, setDelta] = useState([0.01]);
  const [scoreTemperature, setScoreTemperature] = useState([6.0]);
  const [stickinessResult, setStickinessResult] = useState<any>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState([0.7]);
  const [expandedVisualization, setExpandedVisualization] = useState<'force' | 'heatmap' | null>(null);

  // ── Chunk Stickiness – File Input ──────────────────────────────────────────
  const [tempStickinessPath, setTempStickinessPath] = useState('');
  const [stickinessOutputPath, setStickinessOutputPath] = useState('');
  const [stickinessMaxEvalChunks, setStickinessMaxEvalChunks] = useState('-1');
  const [stickinessFilePaths, setStickinessFilePaths] = useState<string[]>([]);

  // ── Chunk Stickiness – Model Config (shared) ───────────────────────────────
  const [stickinessConfigOpen, setStickinessConfigOpen] = useState(false);
  const [stickinessModelPath, setStickinessModelPath] = useState('');
  const [stickinessUseVllm, setStickinessUseVllm] = useState(false);
  const [stickinessVllmApiBase, setStickinessVllmApiBase] = useState('http://localhost:8005/v1');
  const [stickinessVllmModelName, setStickinessVllmModelName] = useState('');

  // ── Retrieval Eval ─────────────────────────────────────────────────────────
  const [retrievalDataJson, setRetrievalDataJson] = useState('');
  const [retrievalResult, setRetrievalResult] = useState<any>(null);
  const [retrievalCuts, setRetrievalCuts] = useState('1,3,5,10');
  const [retrievalSkipEmptyGold, setRetrievalSkipEmptyGold] = useState(true);
  const [tempRetrievalPath, setTempRetrievalPath] = useState('');
  const [retrievalOutputPath, setRetrievalOutputPath] = useState('');
  const [retrievalFilePaths, setRetrievalFilePaths] = useState<string[]>([]);
  const [selectedRetrievalMetric, setSelectedRetrievalMetric] = useState<string | null>(null);
  const [retrievalFamilyExpanded, setRetrievalFamilyExpanded] = useState(false);

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

  useEffect(() => {
    if (location.search.includes('section=retrieval')) {
      setActiveTab('retrieval');
    } else if (activeTab === 'retrieval') {
      setActiveTab('quality');
    }
  }, [location.search, activeTab]);

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
      const maxEvalChunks = parseMaxEvalChunksInput(qualityMaxEvalChunks);
      const data = buildQualityData({
        input_path: qualityFilePaths[0],
        output_path: qualityOutputPath.trim() || undefined,
        max_eval_chunks: maxEvalChunks,
        enable_semantic_similarity: enableSemanticSimilarity,
        enable_boundary_clarity: enableBoundaryClarity,
      });
      const response = await api.chunkQualityFile(data);
      if (response.success) {
        setQualityResult(response.data);
        toast.success(
          qualityOutputPath.trim()
            ? `Chunk quality evaluation completed. Results saved to ${qualityOutputPath.trim()}`
            : 'Chunk quality evaluation completed'
        );
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error((error as Error).message);
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
      const data = buildStickinessData({
        chunks,
        threshold: threshold[0],
        delta: delta[0],
        score_temperature: scoreTemperature[0],
      });
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
      const maxEvalChunks = parseMaxEvalChunksInput(stickinessMaxEvalChunks);
      const data = buildStickinessData({
        input_path: stickinessFilePaths[0],
        output_path: stickinessOutputPath.trim() || undefined,
        max_eval_chunks: maxEvalChunks,
        threshold: threshold[0],
        delta: delta[0],
        score_temperature: scoreTemperature[0],
      });
      const response = await api.chunkStickinessFile(data);
      if (response.success) {
        setStickinessResult(response.data);
        toast.success(
          stickinessOutputPath.trim()
            ? `Stickiness evaluation completed. Results saved to ${stickinessOutputPath.trim()}`
            : 'Stickiness evaluation completed'
        );
      } else {
        toast.error('Evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error((error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const parseCuts = () => {
    const values = retrievalCuts
      .split(',')
      .map((x) => Number(x.trim()))
      .filter((x) => Number.isFinite(x) && x > 0);
    return values.length ? values : [1, 3, 5, 10];
  };

  const handleRetrievalEval = async () => {
    if (!retrievalDataJson.trim()) {
      toast.error('Please enter retrieval eval data in JSON format');
      return;
    }

    setLoading(true);
    try {
      const retrievalData = JSON.parse(retrievalDataJson);
      const response = await api.retrievalEval({
        test: retrievalData,
        cuts: parseCuts(),
        skip_empty_gold: retrievalSkipEmptyGold,
      });

      if (response.success) {
        setRetrievalResult(response.data);
        toast.success('Retrieval evaluation completed');
      } else {
        toast.error('Retrieval evaluation failed: ' + response.message);
      }
    } catch (error) {
      toast.error('Invalid JSON format: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrievalFileEval = async () => {
    if (!retrievalFilePaths.length) {
      toast.error('Please add at least one file path');
      return;
    }

    setLoading(true);
    try {
      const response = await api.retrievalEvalFile({
        input_path: retrievalFilePaths[0],
        output_path: retrievalOutputPath.trim() || undefined,
        cuts: parseCuts(),
        skip_empty_gold: retrievalSkipEmptyGold,
      });

      if (response.success) {
        setRetrievalResult(response.data);
        toast.success(
          retrievalOutputPath.trim()
            ? `Retrieval file evaluation completed. Result saved to ${retrievalOutputPath.trim()}`
            : 'Retrieval file evaluation completed'
        );
      } else {
        toast.error('Retrieval file evaluation failed: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Derived data ────────────────────────────────────────────────────────────
  const qualityChartData =
    qualityResult?.details?.map((item: any, index: number) => ({
      name: `Chunk ${index}`,
      'Boundary Clarity': item.boundary_clarity,
      'Semantic Dissimilarity': item.semantic_dissimilarity,
    })) || [];

  const forceGraphData = useMemo(() => {
    if (!stickinessResult?.graph_complete) return null;
    const nodes: any[] = [];
    const links: any[] = [];
    const graph = stickinessResult.graph_complete;
    const nodeIds = Object.keys(graph);
    nodeIds.forEach((id) => nodes.push({ id, name: `Chunk ${id}` }));
    nodeIds.forEach((source) => {
      Object.entries(graph[source] || {}).forEach(([target, weight]: [string, any]) => {
        const edgeValue = Number(weight);
        if (source !== target && edgeValue >= similarityThreshold[0]) {
          links.push({ source, target, edgeValue, value: edgeValue });
        }
      });
    });
    return { nodes, links };
  }, [stickinessResult, similarityThreshold]);

  const maxForceEdgeValue = useMemo(() => {
    if (!stickinessResult?.graph_complete) return 0;
    let maxValue = 0;
    Object.entries(stickinessResult.graph_complete).forEach(([source, row]: [string, any]) => {
      Object.entries(row || {}).forEach(([target, weight]: [string, any]) => {
        if (source === target) return;
        maxValue = Math.max(maxValue, Number(weight) || 0);
      });
    });
    return maxValue;
  }, [stickinessResult]);

  const getForceLinkColor = useCallback((value: number) => {
    const normalized = maxForceEdgeValue > 0 ? Math.max(0, Math.min(1, value / maxForceEdgeValue)) : 0;
    if (normalized > 0.85) return 'rgba(180, 83, 9, 0.96)';
    if (normalized > 0.65) return 'rgba(239, 68, 68, 0.92)';
    if (normalized > 0.45) return 'rgba(249, 115, 22, 0.88)';
    if (normalized > 0.25) return 'rgba(250, 204, 21, 0.8)';
    return 'rgba(148, 163, 184, 0.5)';
  }, [maxForceEdgeValue]);

  const heatmapData = useMemo(() => {
    if (!stickinessResult?.graph_complete) return [];
    const graph = stickinessResult.graph_complete;
    const nodeIds = Object.keys(graph).sort((a, b) => parseInt(a) - parseInt(b));
    const data: any[] = [];
    nodeIds.forEach((i) => {
      nodeIds.forEach((j) => {
        const dissimilarity = graph[i]?.[j] ?? 1;
        const ii = parseInt(i);
        const jj = parseInt(j);
        data.push({
          x: ii,
          y: jj,
          dissimilarity,
          aboveThreshold: ii !== jj && dissimilarity > threshold[0],
        });
      });
    });
    return data;
  }, [stickinessResult, threshold]);

  const retrievalMetricsByCut = useMemo(() => {
    const agg = retrievalResult?.aggregated;
    if (!agg || typeof agg !== 'object') return [] as Array<{ cut: string; metrics: Record<string, number> }>;
    const grouped: Record<string, Record<string, number>> = {};
    Object.entries(agg).forEach(([key, val]) => {
      const normalizedKey = key.toLowerCase();
      let metric: string | null = null;
      let cut: string | null = null;
      let match = normalizedKey.match(/^p_(\d+)$/);
      if (match) {
        metric = 'precision';
        cut = match[1];
      }
      match = match ?? normalizedKey.match(/^recall_(\d+)$/);
      if (match && !metric) {
        metric = 'recall';
        cut = match[1];
      }
      match = match ?? normalizedKey.match(/^ndcg_cut_(\d+)$/);
      if (match && !metric) {
        metric = 'ndcg';
        cut = match[1];
      }
      match = match ?? normalizedKey.match(/^(map|mrr|ndcg|recall|precision)(?:@|_at_?)(\d+)$/);
      if (match && !metric) {
        metric = match[1].toLowerCase();
        cut = match[2];
      }
      if (!metric || !cut) return;
      if (!grouped[cut]) grouped[cut] = {};
      grouped[cut][metric] = Number(val);
    });
    return Object.keys(grouped)
      .sort((a, b) => Number(a) - Number(b))
      .map((cut) => ({ cut, metrics: grouped[cut] }));
  }, [retrievalResult]);

  const retrievalOverviewMetrics = useMemo(() => {
    const agg = retrievalResult?.aggregated;
    if (!agg || typeof agg !== 'object') return [] as Array<{ key: string; display: string; value: number; metricKey: string }>;
    const items: Array<{ key: string; display: string; value: number; metricKey: string }> = [];
    const pushIfNumber = (rawKey: string, display: string, metricKey: string) => {
      const rawValue = (agg as Record<string, unknown>)[rawKey];
      if (typeof rawValue === 'number') {
        items.push({ key: rawKey, display, value: rawValue, metricKey });
      }
    };
    pushIfNumber('map', 'MAP', 'map');
    pushIfNumber('recip_rank', 'MRR', 'recip_rank');
    pushIfNumber('Rprec', 'R-Precision', 'rprec');
    pushIfNumber('ndcg', 'nDCG', 'ndcg');
    return items;
  }, [retrievalResult]);

  const retrievalCutSummaries = useMemo(() => {
    return retrievalMetricsByCut.map(({ cut, metrics }) => ({ cut, metrics }));
  }, [retrievalMetricsByCut]);

  const addPendingPath = (
    value: string,
    clearValue: (nextValue: string) => void,
    updatePaths: (updater: (prev: string[]) => string[]) => void,
  ) => {
    const trimmedValue = value.trim();
    if (!trimmedValue) return;
    updatePaths((prev) => [...prev, trimmedValue]);
    clearValue('');
  };

  const removePathAtIndex = (
    index: number,
    updatePaths: (updater: (prev: string[]) => string[]) => void,
  ) => {
    updatePaths((prev) => prev.filter((_, itemIndex) => itemIndex !== index));
  };

  const renderSelectedPathList = (paths: string[], onRemove: (index: number) => void) => {
    if (!paths.length) return null;

    return (
      <ScrollArea className="mt-2 h-20 rounded-md border border-slate-200 bg-slate-50">
        <div className="space-y-1 p-2">
          {paths.map((path, idx) => (
            <div key={`${path}-${idx}`} className="flex items-center justify-between gap-2 rounded border border-slate-200 bg-white p-1 px-2">
              <span className="flex-1 truncate font-mono text-xs">{path}</span>
              <Button
                type="button"
                onClick={() => onRemove(idx)}
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
    );
  };

  const renderMetricInfoDialog = (
    selectedMetric: string | null,
    onOpenChange: (open: boolean) => void,
    metricInfoMap: Record<string, ChunkMetricDoc>,
    fallbackSources: Array<{ label: string; url: string }>,
    palette: {
      formulaCard: string;
      formulaText: string;
      exampleCard: string;
      exampleText: string;
      exampleBodyText: string;
      sourceChip: string;
    },
  ) => {
    const metric = selectedMetric ? metricInfoMap[selectedMetric] : null;
    const sources = metric?.sources ?? fallbackSources;
    const interpretationItems = Array.isArray(metric?.interpretation)
      ? metric.interpretation
      : metric?.interpretation
        ? [metric.interpretation]
        : [];

    return (
      <Dialog open={!!selectedMetric} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-[560px] max-h-[85vh] overflow-y-auto">
          {metric && (
            <>
              <DialogHeader>
                <DialogTitle>{metric.title}</DialogTitle>
                <DialogDescription>{metric.blurb}</DialogDescription>
              </DialogHeader>
              <div className="min-w-0 space-y-4">
                <div className={`${palette.formulaCard} min-w-0 overflow-hidden`}>
                  <div className={`mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] ${palette.formulaText}`}>
                    <Sigma className="h-3.5 w-3.5" />
                    Formula (LaTeX)
                  </div>
                  <div className="overflow-x-auto rounded-lg bg-white/80 p-3">
                    <BlockMath math={metric.formulaLatex} />
                  </div>
                </div>
                <div className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-4">
                  <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Interpretation</div>
                  {interpretationItems.length <= 1 ? (
                    <p className="break-words text-sm text-slate-700">{interpretationItems[0]}</p>
                  ) : (
                    <ul className="space-y-1 list-disc break-words pl-5 text-sm text-slate-700">
                      {interpretationItems.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  )}
                </div>
                {metric.variables?.length ? (
                  <div className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white p-4">
                    <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Variables</div>
                    <ul className="space-y-2 text-sm text-slate-700">
                      {metric.variables.map((item, idx) => (
                        <li key={idx} className="overflow-x-auto rounded-lg bg-slate-50 px-3 py-2">
                          <BlockMath math={item} />
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                <div className={`${palette.exampleCard} min-w-0 overflow-hidden`}>
                  <div className={`mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] ${palette.exampleText}`}>
                    <FlaskConical className="h-3.5 w-3.5" />
                    Project-aligned example
                  </div>
                  <ul className={`space-y-1 list-disc break-words pl-5 text-sm ${palette.exampleBodyText}`}>
                    {metric.projectExample.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div className="min-w-0 overflow-hidden rounded-xl border border-slate-200 bg-white p-4">
                  <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Source</div>
                  <div className="flex flex-wrap gap-2">
                    {sources.map((source, idx) => (
                      <a
                        key={idx}
                        href={source.url}
                        target="_blank"
                        rel="noreferrer"
                        className={`rounded-full border px-2 py-1 text-xs ${palette.sourceChip}`}
                      >
                        {source.label}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    );
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Component-Level Evaluation</h1>
          <p className="text-slate-600">
            {isRetrievalSection
              ? 'Evaluate retrieval quality metrics (MAP/MRR/nDCG/Recall@k)'
              : 'Evaluate chunk quality and stickiness with advanced visualizations'}
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'quality' | 'stickiness' | 'retrieval')}>
          <div className="mb-6 space-y-3">
            {!isRetrievalSection ? (
              <div className="inline-flex p-1 rounded-xl bg-indigo-50 border border-indigo-100">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setActiveTab('quality')}
                  className={activeTab === 'quality' ? 'bg-white shadow-sm text-indigo-700' : 'text-indigo-600'}
                >
                  Chunk Quality
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setActiveTab('stickiness')}
                  className={activeTab === 'stickiness' ? 'bg-white shadow-sm text-indigo-700' : 'text-indigo-600'}
                >
                  Chunk Stickiness
                </Button>
              </div>
            ) : (
              <div className="inline-flex p-1 rounded-xl bg-cyan-50 border border-cyan-100">
                <span className="px-3 py-1.5 text-sm font-medium text-cyan-700">Retrieval Evaluation</span>
              </div>
            )}
          </div>

          {/* ── Chunk Quality Tab ─────────────────────────────────────────── */}
          {!isRetrievalSection && (
          <TabsContent value="quality">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input column */}
              <div className="lg:h-[860px]">
                <ScrollArea className="h-full lg:pr-2">
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
                        onKeyDown={(e) => fillPlaceholderOnTab(e, qualityChunksJson, e.currentTarget.placeholder, setQualityChunksJson)}
                        placeholder={chunkJsonPlaceholder}
                        className="h-[220px] resize-none overflow-y-auto font-mono text-sm"
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
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempQualityPath, e.currentTarget.placeholder, setTempQualityPath)}
                          placeholder="/path/to/chunks.json (enter server path and click +)"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              addPendingPath(tempQualityPath, setTempQualityPath, setQualityFilePaths);
                            }
                          }}
                        />
                        <PathPickerButton
                          mode="file"
                          value={tempQualityPath}
                          allowedExtensions={['.json']}
                          title="Select Chunk Quality Eval File"
                          description="This browser reads the filesystem on the machine running the backend service."
                          onSelect={(path) => setQualityFilePaths((prev) => [...prev, path])}
                        />
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            if (tempQualityPath.trim()) {
                              addPendingPath(tempQualityPath, setTempQualityPath, setQualityFilePaths);
                            }
                          }}
                        >
                          <Plus className="w-4 h-4" />
                        </Button>
                      </div>
                      {renderSelectedPathList(qualityFilePaths, (idx) => removePathAtIndex(idx, setQualityFilePaths))}
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
                    <div>
                      <Label>Output Results JSON Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={qualityOutputPath}
                          onChange={(e) => setQualityOutputPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, qualityOutputPath, e.currentTarget.placeholder, setQualityOutputPath)}
                          placeholder="/path/to/chunk_quality_results.json"
                        />
                        <PathPickerButton
                          mode="directory"
                          value={qualityOutputPath}
                          title="Select Output Directory"
                          description="Pick the folder to write the result file into, then edit the filename if needed."
                          onSelect={(selectedDirectory) => {
                            const currentName = qualityOutputPath.split(/[\\/]/).filter(Boolean).at(-1);
                            const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                            setQualityOutputPath(
                              currentName && currentName.includes('.')
                                ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                                : selectedDirectory,
                            );
                          }}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        Optional. If provided, the backend will write the complete JSON shown on the right to this file.
                      </p>
                    </div>
                    <div>
                      <Label>Max Eval Chunks</Label>
                      <Input
                        value={qualityMaxEvalChunks}
                        onChange={(e) => setQualityMaxEvalChunks(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, qualityMaxEvalChunks, e.currentTarget.placeholder, setQualityMaxEvalChunks)}
                        placeholder="-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Limit file evaluation to the first N chunks. Use <code>-1</code> to evaluate all chunks.
                      </p>
                    </div>
                    <Button onClick={handleQualityFileEval} disabled={loading} className="w-full" variant="outline">
                      {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</> : 'Evaluate from File'}
                    </Button>
                  </div>
                </Card>
                </div>
                </ScrollArea>
              </div>

              {/* Results column */}
              <Card className="flex h-[860px] flex-col p-6">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h2 className="font-bold">Quality Results</h2>
                    <p className="text-xs text-slate-500 mt-1">Click BC/DS cards to open paper-style metric notes.</p>
                  </div>
                  {renderMetricInfoDialog(
                    selectedChunkMetric,
                    (open) => !open && setSelectedChunkMetric(null),
                    chunkMetricInfo,
                    chunkMetricSources,
                    {
                      formulaCard: 'rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 via-cyan-50 to-white p-4 shadow-sm',
                      formulaText: 'text-blue-700',
                      exampleCard: 'rounded-xl border border-blue-100 bg-blue-50/60 p-4',
                      exampleText: 'text-blue-700',
                      exampleBodyText: 'text-blue-900',
                      sourceChip: 'border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100',
                    },
                  )}
                </div>
                <div className="min-h-0 flex-1">
                {loading ? (
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin" />
                  </div>
                ) : qualityResult ? (
                  <ScrollArea className="h-full">
                    <div className="space-y-6">
                      <div className="grid grid-cols-2 gap-4">
                        <button
                          type="button"
                          onClick={() => setSelectedChunkMetric('bc')}
                          className="p-4 text-left bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-blue-300"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="text-sm text-slate-600 mb-1">Avg Boundary Clarity</div>
                            <span className="rounded-full border border-blue-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-blue-700">BC</span>
                          </div>
                          <div className="text-2xl font-bold text-blue-900">{qualityResult.avg_boundary_clarity?.toFixed(4) || '-'}</div>
                          <p className="mt-2 text-xs text-slate-600">Perplexity-ratio based boundary metric from MoC-style chunk evaluation.</p>
                        </button>
                        <button
                          type="button"
                          onClick={() => setSelectedChunkMetric('ds')}
                          className="p-4 text-left bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-purple-300"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="text-sm text-slate-600 mb-1">Avg Semantic Dissimilarity</div>
                            <span className="rounded-full border border-purple-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-purple-700">DS</span>
                          </div>
                          <div className="text-2xl font-bold text-purple-900">{qualityResult.avg_semantic_dissimilarity?.toFixed(4) || '-'}</div>
                          <p className="mt-2 text-xs text-slate-600">Embedding cosine-separation metric over adjacent chunk pairs.</p>
                        </button>
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
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400">
                    <div className="text-center">
                      <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Evaluation results will appear here</p>
                    </div>
                  </div>
                )}
                </div>
              </Card>
            </div>
          </TabsContent>
          )}

          {/* ── Chunk Stickiness Tab ──────────────────────────────────────── */}
          {!isRetrievalSection && (
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
                        onKeyDown={(e) => fillPlaceholderOnTab(e, stickinessChunksJson, e.currentTarget.placeholder, setStickinessChunksJson)}
                        placeholder={chunkJsonPlaceholder}
                        className="h-[220px] resize-none overflow-y-auto font-mono text-sm"
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
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempStickinessPath, e.currentTarget.placeholder, setTempStickinessPath)}
                          placeholder="/path/to/chunks.json (enter server path and click +)"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              addPendingPath(tempStickinessPath, setTempStickinessPath, setStickinessFilePaths);
                            }
                          }}
                        />
                        <PathPickerButton
                          mode="file"
                          value={tempStickinessPath}
                          allowedExtensions={['.json']}
                          title="Select Chunk Stickiness Eval File"
                          description="This browser reads the filesystem on the machine running the backend service."
                          onSelect={(path) => setStickinessFilePaths((prev) => [...prev, path])}
                        />
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            if (tempStickinessPath.trim()) {
                              addPendingPath(tempStickinessPath, setTempStickinessPath, setStickinessFilePaths);
                            }
                          }}
                        >
                          <Plus className="w-4 h-4" />
                        </Button>
                      </div>
                      {renderSelectedPathList(stickinessFilePaths, (idx) => removePathAtIndex(idx, setStickinessFilePaths))}
                    </div>
                    <div>
                      <Label>Output Results JSON Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={stickinessOutputPath}
                          onChange={(e) => setStickinessOutputPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, stickinessOutputPath, e.currentTarget.placeholder, setStickinessOutputPath)}
                          placeholder="/path/to/chunk_stickiness_results.json"
                        />
                        <PathPickerButton
                          mode="directory"
                          value={stickinessOutputPath}
                          title="Select Output Directory"
                          description="Pick the folder to write the result file into, then edit the filename if needed."
                          onSelect={(selectedDirectory) => {
                            const currentName = stickinessOutputPath.split(/[\\/]/).filter(Boolean).at(-1);
                            const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                            setStickinessOutputPath(
                              currentName && currentName.includes('.')
                                ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                                : selectedDirectory,
                            );
                          }}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        Optional. If provided, the backend will write the complete JSON shown below to this file.
                      </p>
                    </div>
                    <div>
                      <Label>Max Eval Chunks</Label>
                      <Input
                        value={stickinessMaxEvalChunks}
                        onChange={(e) => setStickinessMaxEvalChunks(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, stickinessMaxEvalChunks, e.currentTarget.placeholder, setStickinessMaxEvalChunks)}
                        placeholder="-1"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Limit file evaluation to the first N chunks. Use <code>-1</code> to evaluate all chunks.
                      </p>
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
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Threshold</Label>
                        <span className="text-sm font-mono text-slate-600">{threshold[0].toFixed(2)}</span>
                      </div>
                      <Slider
                        value={threshold}
                        onValueChange={setThreshold}
                        min={0.5}
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
                        max={0.2}
                        step={0.01}
                        disabled={loading}
                      />
                      <p className="text-xs text-slate-500">Higher values favor adjacent chunks</p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>BC Temperature</Label>
                        <span className="text-sm font-mono text-slate-600">{scoreTemperature[0].toFixed(1)}</span>
                      </div>
                      <Slider
                        value={scoreTemperature}
                        onValueChange={setScoreTemperature}
                        min={1}
                        max={12}
                        step={0.5}
                        disabled={loading}
                      />
                      <p className="text-xs text-slate-500">Higher values stretch the high-score region of BC mapping</p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Visualization Threshold</Label>
                        <span className="text-sm font-mono text-slate-600">{similarityThreshold[0].toFixed(2)}</span>
                      </div>
                      <Slider
                        value={similarityThreshold}
                        onValueChange={setSimilarityThreshold}
                        min={0.5}
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
                  {renderMetricInfoDialog(
                    selectedChunkMetric,
                    (open) => !open && setSelectedChunkMetric(null),
                    chunkMetricInfo,
                    chunkMetricSources,
                    {
                      formulaCard: 'rounded-2xl border border-emerald-200 bg-gradient-to-br from-emerald-50 via-teal-50 to-white p-4 shadow-sm',
                      formulaText: 'text-emerald-700',
                      exampleCard: 'rounded-xl border border-emerald-100 bg-emerald-50/60 p-4',
                      exampleText: 'text-emerald-700',
                      exampleBodyText: 'text-emerald-900',
                      sourceChip: 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100',
                    },
                  )}
                  {/* Entropy Values */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <button
                      type="button"
                      onClick={() => setSelectedChunkMetric('cs')}
                      className="md:col-span-2 p-5 text-left rounded-xl border border-emerald-200 bg-gradient-to-br from-emerald-50 via-teal-50 to-white hover:shadow-md hover:border-emerald-300 transition-all"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-700">Chunk Graph Metric</div>
                          <div className="text-base font-semibold text-slate-800 mt-1">Chunk Stickiness (CS)</div>
                          <p className="text-xs text-slate-600 mt-1">Structural-entropy based global cohesion metric from MoC-style relation evaluation.</p>
                        </div>
                        <span className="rounded-full border border-emerald-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-emerald-700">CS Formula</span>
                      </div>
                    </button>

                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="p-2 bg-green-100 rounded-lg">
                          <Cpu className="w-5 h-5 text-green-700" />
                        </div>
                        <div>
                          <h3 className="font-bold">Normalized Complete Graph Entropy</h3>
                          <p className="text-xs text-slate-600">All chunks fully connected, normalized by log2(active nodes)</p>
                        </div>
                      </div>
                      <div className="text-4xl font-bold text-green-900 mb-2">
                        {stickinessResult.normalized_structural_entropy_complete?.toFixed(6) || '-'}
                      </div>
                      <p className="text-sm text-slate-600">Lower values indicate stronger overall cohesion</p>
                    </Card>
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="p-2 bg-orange-100 rounded-lg">
                          <Cpu className="w-5 h-5 text-orange-700" />
                        </div>
                        <div>
                          <h3 className="font-bold">Normalized Incomplete Graph Entropy</h3>
                          <p className="text-xs text-slate-600">After threshold filtering, normalized by log2(active nodes)</p>
                        </div>
                      </div>
                      <div className="text-4xl font-bold text-orange-900 mb-2">
                        {stickinessResult.normalized_structural_entropy_incomplete?.toFixed(6) || '-'}
                      </div>
                      <p className="text-sm text-slate-600">Difference from complete shows threshold impact</p>
                    </Card>
                  </div>

                  {/* Visualizations */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    {/* Force Graph – explicit size to prevent overflow */}
                    <Card className="p-6 overflow-hidden xl:col-span-2">
                      <div className="mb-4 flex items-start justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <Network className="w-5 h-5" />
                          <h3 className="font-bold">Chunk Dependency Graph</h3>
                        </div>
                        <Button type="button" variant="outline" size="sm" onClick={() => setExpandedVisualization('force')}>
                          <Maximize2 className="mr-2 h-4 w-4" />
                          Expand
                        </Button>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Force-directed layout – shows normalized graph edges whose value exceeds the visualization threshold
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
                            linkWidth={(link: any) => Math.max(1.2, Math.pow(link.value, 1.15) * 8)}
                            linkColor={(link: any) => getForceLinkColor(link.value)}
                            nodeRelSize={6}
                            linkDirectionalParticles={2}
                            linkDirectionalParticleWidth={(link: any) => link.value * 4}
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center">
                            <div className="text-center text-slate-400">
                              <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
                              <p>No edges above visualization threshold</p>
                              <p className="text-xs">Current max edge value: {maxForceEdgeValue.toFixed(3)}</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </Card>

                    {/* Heatmap */}
                    <Card className="p-6">
                      <div className="mb-4 flex items-start justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <Grid3x3 className="w-5 h-5" />
                          <h3 className="font-bold">Edge Value Heatmap</h3>
                        </div>
                        <Button type="button" variant="outline" size="sm" onClick={() => setExpandedVisualization('heatmap')}>
                          <Maximize2 className="mr-2 h-4 w-4" />
                          Expand
                        </Button>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Matrix view of normalized graph edge values returned by the backend
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
                                className="relative aspect-square rounded-sm border border-slate-200"
                                style={{ backgroundColor: `rgba(239, 68, 68, ${Math.max(0.12, Math.min(1, cell.dissimilarity))})` }}
                                title={`Chunk ${cell.x} -> ${cell.y}: edgeValue=${cell.dissimilarity.toFixed(3)}${cell.aboveThreshold ? `, above threshold ${threshold[0].toFixed(2)}` : ''}`}
                              >
                                {cell.aboveThreshold ? (
                                  <span className="absolute right-1 top-1 h-2.5 w-2.5 rounded-full border border-white/80 bg-slate-950/80" />
                                ) : null}
                                <span className="absolute inset-x-0 bottom-0 truncate px-1 pb-0.5 text-center text-[10px] font-medium text-white/90 mix-blend-plus-lighter">
                                  {cell.dissimilarity.toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                          <div className="flex items-center justify-between mt-4 text-xs">
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-rose-100 rounded border border-slate-200" />
                              Low edge value
                            </span>
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-rose-600 rounded border border-slate-200" />
                              High edge value
                            </span>
                            <span className="flex items-center gap-2">
                              <div className="w-2.5 h-2.5 rounded-full border border-white/80 bg-slate-950/80" />
                              Above threshold
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

                    <Dialog open={expandedVisualization !== null} onOpenChange={(open) => !open && setExpandedVisualization(null)}>
                    <DialogContent className="w-[min(96vw,88rem)] max-w-[88rem] max-h-[92vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle>
                          {expandedVisualization === 'force' ? 'Chunk Dependency Graph' : 'Edge Value Heatmap'}
                        </DialogTitle>
                        <DialogDescription>
                          {expandedVisualization === 'force'
                            ? 'Expanded view of normalized graph edges above the visualization threshold.'
                            : 'Expanded matrix view of normalized graph edge values.'}
                        </DialogDescription>
                      </DialogHeader>
                      {expandedVisualization === 'force' ? (
                        <div className="space-y-4">
                          <div className="rounded-lg border border-slate-200 bg-slate-50 overflow-hidden" style={{ height: 680 }}>
                            {forceGraphData && forceGraphData.nodes.length > 0 ? (
                              <ForceGraph2D
                                graphData={forceGraphData}
                                width={1200}
                                height={680}
                                nodeLabel="name"
                                nodeAutoColorBy="id"
                            linkWidth={(link: any) => Math.max(1.4, Math.pow(link.value, 1.15) * 10)}
                            linkColor={(link: any) => getForceLinkColor(link.value)}
                                nodeRelSize={7}
                                linkDirectionalParticles={2}
                                linkDirectionalParticleWidth={(link: any) => link.value * 4}
                              />
                            ) : (
                              <div className="flex h-full items-center justify-center text-center text-slate-400">
                                <div>
                                  <Network className="mx-auto mb-2 h-12 w-12 opacity-50" />
                                  <p>No edges above visualization threshold</p>
                                  <p className="text-xs">Current max edge value: {maxForceEdgeValue.toFixed(3)}</p>
                                </div>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-2">
                              <div className="h-1 w-8 rounded-full bg-slate-300" />
                              Lower edge value
                            </span>
                            <span className="text-slate-500">Current max edge value: {maxForceEdgeValue.toFixed(3)}</span>
                            <span className="flex items-center gap-2">
                              <div className="h-2 w-8 rounded-full bg-orange-700" />
                              Higher edge value
                            </span>
                          </div>
                        </div>
                      ) : heatmapData.length > 0 ? (
                        <div className="space-y-4">
                          <ScrollArea className="h-[720px]">
                            <div
                              className="grid gap-1"
                              style={{
                                gridTemplateColumns: `repeat(${Math.round(Math.sqrt(heatmapData.length))}, minmax(0, 1fr))`,
                              }}
                            >
                              {heatmapData.map((cell, idx) => (
                                <div
                                  key={`expanded-${idx}`}
                                  className="relative aspect-square rounded-sm border border-slate-200"
                                  style={{ backgroundColor: `rgba(239, 68, 68, ${Math.max(0.12, Math.min(1, cell.dissimilarity))})` }}
                                  title={`Chunk ${cell.x} -> ${cell.y}: edgeValue=${cell.dissimilarity.toFixed(3)}${cell.aboveThreshold ? `, above threshold ${threshold[0].toFixed(2)}` : ''}`}
                                >
                                  {cell.aboveThreshold ? (
                                    <span className="absolute right-1 top-1 h-2.5 w-2.5 rounded-full border border-white/80 bg-slate-950/80" />
                                  ) : null}
                                  <span className="absolute inset-x-0 bottom-0 truncate px-1 pb-0.5 text-center text-[10px] font-medium text-white/90 mix-blend-plus-lighter">
                                    {cell.dissimilarity.toFixed(2)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </ScrollArea>
                          <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-rose-100 rounded border border-slate-200" />
                              Low edge value
                            </span>
                            <span className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-rose-600 rounded border border-slate-200" />
                              High edge value
                            </span>
                            <span className="flex items-center gap-2">
                              <div className="w-2.5 h-2.5 rounded-full border border-white/80 bg-slate-950/80" />
                              Above threshold
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="py-12 text-center text-slate-400">
                          <Grid3x3 className="mx-auto mb-2 h-12 w-12 opacity-50" />
                          <p>No heatmap data available</p>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>

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
          )}

          {/* ── Retrieval Evaluation Tab ─────────────────────────────────── */}
          {isRetrievalSection && (
          <TabsContent value="retrieval">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="lg:h-[860px]">
                <ScrollArea className="h-full lg:pr-2">
                <div className="space-y-6">
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-indigo-600" />
                    Direct JSON Input
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Retrieval Eval Data (JSON)</Label>
                      <Textarea
                        value={retrievalDataJson}
                        onChange={(e) => setRetrievalDataJson(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, retrievalDataJson, e.currentTarget.placeholder, setRetrievalDataJson)}
                        placeholder={'[\n  {\n    "_id": "q1",\n    "rag_retrieval": [{"doc_id":"d1","chunk_id":"0","text":"...","retrieval_score":0.92}],\n    "gold_reference": [{"doc_id":"d1","chunk_id":"0","text":"..."}]\n  }\n]'}
                        className="h-[280px] resize-none overflow-y-auto font-mono text-xs"
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <Label>Cut Points (comma separated)</Label>
                        <Input
                          value={retrievalCuts}
                          onChange={(e) => setRetrievalCuts(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, retrievalCuts, e.currentTarget.placeholder, setRetrievalCuts)}
                          placeholder="1,3,5"
                        />
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg self-end">
                        <div>
                          <Label className="text-sm">Skip Empty Gold</Label>
                          <p className="text-xs text-slate-500">Ignore rows without gold_reference</p>
                        </div>
                        <Switch checked={retrievalSkipEmptyGold} onCheckedChange={setRetrievalSkipEmptyGold} />
                      </div>
                    </div>

                    <Button onClick={handleRetrievalEval} disabled={loading} className="w-full">
                      {loading ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</>) : ('Start Retrieval Evaluation')}
                    </Button>
                  </div>
                </Card>

                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    File Evaluation
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Evaluation Data File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempRetrievalPath}
                          onChange={(e) => setTempRetrievalPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempRetrievalPath, e.currentTarget.placeholder, setTempRetrievalPath)}
                          placeholder="/path/to/retrieval_eval_data.json (enter server path and click +)"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              addPendingPath(tempRetrievalPath, setTempRetrievalPath, setRetrievalFilePaths);
                            }
                          }}
                        />
                        <PathPickerButton
                          mode="file"
                          value={tempRetrievalPath}
                          allowedExtensions={['.json']}
                          title="Select Retrieval Eval File"
                          description="This browser reads the filesystem on the machine running the backend service."
                          onSelect={(path) => setRetrievalFilePaths((prev) => [...prev, path])}
                        />
                        <Button
                          type="button"
                          onClick={() => {
                            if (tempRetrievalPath.trim()) {
                              addPendingPath(tempRetrievalPath, setTempRetrievalPath, setRetrievalFilePaths);
                            }
                          }}
                          size="sm"
                          variant="outline"
                        >
                          <Plus className="w-4 h-4" />
                        </Button>
                      </div>
                      {renderSelectedPathList(retrievalFilePaths, (idx) => removePathAtIndex(idx, setRetrievalFilePaths))}
                    </div>

                    <div>
                      <Label>Output Summary JSON Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={retrievalOutputPath}
                          onChange={(e) => setRetrievalOutputPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, retrievalOutputPath, e.currentTarget.placeholder, setRetrievalOutputPath)}
                          placeholder="/path/to/retrieval_eval_summary.json"
                        />
                        <PathPickerButton
                          mode="directory"
                          value={retrievalOutputPath}
                          title="Select Output Directory"
                          description="Pick the folder to write the summary file into, then edit the filename if needed."
                          onSelect={(selectedDirectory) => {
                            const currentName = retrievalOutputPath.split(/[\\/]/).filter(Boolean).at(-1);
                            const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                            setRetrievalOutputPath(
                              currentName && currentName.includes('.')
                                ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                                : selectedDirectory,
                            );
                          }}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">Optional. If provided, backend will save retrieval evaluation result JSON to this path.</p>
                    </div>

                    <Button onClick={handleRetrievalFileEval} disabled={loading} className="w-full" variant="outline">
                      {loading ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Evaluating...</>) : ('Evaluate Retrieval from File')}
                    </Button>
                  </div>
                </Card>
                </div>
                </ScrollArea>
              </div>

              <Card className="flex h-[860px] flex-col p-6">
                <h2 className="font-bold mb-4">Retrieval Results</h2>
                <div className="min-h-0 flex-1">
                {loading ? (
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400"><Loader2 className="w-8 h-8 animate-spin" /></div>
                ) : retrievalResult ? (
                  <ScrollArea className="h-full">
                    <div className="space-y-6">
                      {retrievalResult.aggregated && (
                        <div>
                          <div className="flex items-start justify-between gap-4 mb-3">
                            <div>
                              <h3 className="text-sm font-medium">Aggregated Metrics</h3>
                              <p className="text-xs text-slate-500 mt-1">
                                Summary cards at a key cut-off, with expandable panels for detailed @k breakdowns.
                              </p>
                            </div>
                            {renderMetricInfoDialog(
                              selectedRetrievalMetric,
                              (open) => !open && setSelectedRetrievalMetric(null),
                              retrievalMetricInfo,
                              [],
                              {
                                formulaCard:
                                  'rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-cyan-50 to-white p-4 shadow-sm',
                                formulaText: 'text-indigo-700',
                                exampleCard: 'rounded-xl border border-indigo-100 bg-indigo-50/60 p-4',
                                exampleText: 'text-indigo-700',
                                exampleBodyText: 'text-indigo-900',
                                sourceChip: 'border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100',
                              },
                            )}
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            {retrievalOverviewMetrics.map(({ key, display, value, metricKey }) => {
                              const metricDoc = retrievalMetricInfo[metricKey];
                              return (
                                <button
                                  key={key}
                                  type="button"
                                  onClick={() => metricDoc && setSelectedRetrievalMetric(metricKey)}
                                  className="p-4 text-left bg-gradient-to-br from-indigo-50 to-cyan-50 rounded-lg border border-indigo-200 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-indigo-300 disabled:cursor-default disabled:hover:translate-y-0 disabled:hover:shadow-none"
                                  disabled={!metricDoc}
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="text-sm text-slate-600 mb-1">{display}</div>
                                    {metricDoc && (
                                      <span className="rounded-full border border-indigo-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-indigo-700">
                                        Formula
                                      </span>
                                    )}
                                  </div>
                                  <div className="text-2xl font-bold text-indigo-900">{value.toFixed(4)}</div>
                                  {metricDoc && <p className="mt-2 text-xs leading-5 text-slate-600">{metricDoc.blurb}</p>}
                                </button>
                              );
                            })}

                            {retrievalCutSummaries.length > 0 && (() => {
                              const topCut = retrievalCutSummaries[retrievalCutSummaries.length - 1];
                              return (
                                <div key="retrieval_cut_family" className="col-span-2 rounded-xl border border-indigo-200 bg-gradient-to-br from-slate-50 via-indigo-50 to-cyan-50 p-4 text-left shadow-sm">
                                  <button
                                    type="button"
                                    className="w-full flex items-start justify-between gap-3"
                                    onClick={() => setRetrievalFamilyExpanded((v) => !v)}
                                  >
                                    <div className="flex-1">
                                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Top-k Retrieval Panel</div>
                                      <div className="mt-1 text-sm text-slate-700">Cut Family (@k)</div>
                                      <div className="mt-1 flex items-end gap-4">
                                        <div className="text-2xl font-bold text-indigo-900">@{topCut.cut}</div>
                                        <div className="text-xs text-slate-600 pb-1">Precision {Number(topCut.metrics.precision ?? 0).toFixed(4)} · Recall {Number(topCut.metrics.recall ?? 0).toFixed(4)} · nDCG {Number(topCut.metrics.ndcg ?? 0).toFixed(4)}</div>
                                      </div>
                                      <p className="mt-1 text-xs text-slate-600">Expand to inspect each cut and click metric chips for Precision / Recall / nDCG explanations.</p>
                                    </div>
                                    {retrievalFamilyExpanded ? <ChevronUp className="w-4 h-4 text-indigo-700 mt-1" /> : <ChevronDown className="w-4 h-4 text-indigo-700 mt-1" />}
                                  </button>

                                  <div className="mt-3 rounded-lg border border-indigo-100 bg-white/70 p-3">
                                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-500">Cut profile</div>
                                    <div className={`grid gap-2 ${retrievalCutSummaries.length >= 4 ? 'grid-cols-4' : 'grid-cols-2'}`}>
                                      {retrievalCutSummaries.map(({ cut, metrics }) => {
                                        return (
                                          <button
                                            key={`cut_bar_${cut}`}
                                            type="button"
                                            onClick={() => setRetrievalFamilyExpanded(true)}
                                            className="rounded-md border border-indigo-100 bg-white p-2 text-center hover:border-indigo-300"
                                          >
                                            <div className="mx-auto mb-2 rounded-md border border-indigo-100 bg-indigo-50 px-2 py-1 text-xs font-semibold text-indigo-800">
                                              @{cut}
                                            </div>
                                            <div className="text-[10px] uppercase text-slate-500">@{cut}</div>
                                            <div className="mt-1 flex items-center justify-center gap-1">
                                              <span
                                                role="button"
                                                tabIndex={0}
                                                onClick={(e) => {
                                                  e.stopPropagation();
                                                  setSelectedRetrievalMetric('precision');
                                                }}
                                                onKeyDown={(e) => {
                                                  if (e.key === 'Enter' || e.key === ' ') {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    setSelectedRetrievalMetric('precision');
                                                  }
                                                }}
                                                className="rounded border border-indigo-100 bg-indigo-50/60 px-1.5 py-0.5 text-[10px] text-indigo-700"
                                              >
                                                P {Number(metrics.precision ?? 0).toFixed(2)}
                                              </span>
                                              <span
                                                role="button"
                                                tabIndex={0}
                                                onClick={(e) => {
                                                  e.stopPropagation();
                                                  setSelectedRetrievalMetric('recall');
                                                }}
                                                onKeyDown={(e) => {
                                                  if (e.key === 'Enter' || e.key === ' ') {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    setSelectedRetrievalMetric('recall');
                                                  }
                                                }}
                                                className="rounded border border-indigo-100 bg-indigo-50/60 px-1.5 py-0.5 text-[10px] text-indigo-700"
                                              >
                                                R {Number(metrics.recall ?? 0).toFixed(2)}
                                              </span>
                                              <span
                                                role="button"
                                                tabIndex={0}
                                                onClick={(e) => {
                                                  e.stopPropagation();
                                                  setSelectedRetrievalMetric('ndcg');
                                                }}
                                                onKeyDown={(e) => {
                                                  if (e.key === 'Enter' || e.key === ' ') {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    setSelectedRetrievalMetric('ndcg');
                                                  }
                                                }}
                                                className="rounded border border-indigo-100 bg-indigo-50/60 px-1.5 py-0.5 text-[10px] text-indigo-700"
                                              >
                                                N {Number(metrics.ndcg ?? 0).toFixed(2)}
                                              </span>
                                            </div>
                                          </button>
                                        );
                                      })}
                                    </div>
                                  </div>

                                  {retrievalFamilyExpanded && (
                                    <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                                      {retrievalCutSummaries.map(({ cut, metrics }) => (
                                        <div key={`cut_${cut}`} className="rounded-md border border-indigo-200 bg-white/90 px-3 py-3">
                                          <div className="text-xs text-slate-600 uppercase">@{cut}</div>
                                          <div className="mt-2 grid grid-cols-3 gap-2">
                                            {[
                                              { label: 'Precision', value: metrics.precision, metricKey: 'precision' },
                                              { label: 'Recall', value: metrics.recall, metricKey: 'recall' },
                                              { label: 'nDCG', value: metrics.ndcg, metricKey: 'ndcg' },
                                            ].map((item) => (
                                              <button
                                                key={`${cut}-${item.metricKey}`}
                                                type="button"
                                                onClick={() => setSelectedRetrievalMetric(item.metricKey)}
                                                className="rounded-md border border-indigo-100 bg-indigo-50/40 px-2 py-2 text-left hover:bg-indigo-50"
                                              >
                                                <div className="text-[10px] uppercase text-slate-500">{item.label}@{cut}</div>
                                                <div className="text-sm font-semibold text-indigo-900">{Number(item.value ?? 0).toFixed(4)}</div>
                                              </button>
                                            ))}
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })()}
                          </div>
                        </div>
                      )}
                      <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto">{JSON.stringify(retrievalResult, null, 2)}</pre>
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400"><div className="text-center"><BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" /><p>Retrieval results will appear here</p></div></div>
                )}
                </div>
              </Card>
            </div>
          </TabsContent>
          )}
        </Tabs>
      </div>
    </div>
  );
}

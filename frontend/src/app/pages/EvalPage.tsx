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
import { Loader2, BarChart3, FileText, Sparkles, Settings, FolderOpen, Plus, X, FlaskConical, Sigma, ChevronDown, ChevronUp } from 'lucide-react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { api } from '../utils/api';
import { toast } from 'sonner';

export default function EvalPage() {
  const [loading, setLoading] = useState(false);
  const [selectedTraditionalMetric, setSelectedTraditionalMetric] = useState<string | null>(null);
  const [selectedRagasMetric, setSelectedRagasMetric] = useState<string | null>(null);
  const [bleuExpanded, setBleuExpanded] = useState(false);

  const traditionalPlaceholder = `[
\t{
\t\t"_id": 1,
\t\t"input": "According to the 19th-century writer Baron Ernouf, who was the brother of Waldrada of Lotharingia?",
\t\t"llm_ans": "According to Baron Ernouf, Waldrada was the sister of Thietgaud, bishop of Trier, so her brother was Thietgaud.",
\t\t"answer": "Baron Ernouf suggested that Waldrada was the sister of Thietgaud, the bishop of Trier.",
\t\t"rag_retrieval": [
\t\t\t{
\t\t\t\t"text": "Baron Ernouf suggested that Waldrada was the sister of Thietgaud, the bishop of Trier.",
\t\t\t\t"retrieval_score": 0.6081,
\t\t\t\t"doc_id": "41ac2...",
\t\t\t\t"chunk_id": "2"
\t\t\t}
\t\t],
\t\t"gold_reference": [
\t\t\t{
\t\t\t\t"text": "Baron Ernouf suggested that Waldrada was the sister of Thietgaud, the bishop of Trier.",
\t\t\t\t"doc_id": "41ac2...",
\t\t\t\t"chunk_id": "1"
\t\t\t}
\t\t]
\t}
]`;

  const ragasPlaceholder = `Format 1 - Standard RAGAS:
{
\t"question": ["Q1?", "Q2?"],
\t"answer": ["A1", "A2"],
\t"contexts": [["C1a", "C1b"], ["C2"]],
\t"ground_truth": ["GT1", "GT2"]
}

Format 2 - Current Eval Format:
[
\t{
\t\t"_id": 1,
\t\t"input": "Question?",
\t\t"llm_ans": "LLM answer",
\t\t"answer": "Ground truth",
\t\t"rag_retrieval": [
\t\t\t{ "text": "Context 1", "retrieval_score": 0.91 },
\t\t\t{ "text": "Context 2", "retrieval_score": 0.87 }
\t\t],
\t\t"gold_reference": [
\t\t\t{ "text": "Reference context", "doc_id": "doc_1", "chunk_id": "1" }
\t\t]
\t}
]`;

  const fillPlaceholderOnTab = (
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
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

  type MetricCardDoc = {
    title: string;
    blurb: string;
    formulaLatex: string;
    interpretation: string;
    variables: string[];
    projectExample: string[];
  };

  const traditionalMetricInfo: Record<string, MetricCardDoc> = {
    f1: {
      title: 'F1 Score (Token-level, Max-over-References)',
      blurb: 'Measures overlap between llm_ans and answer, balancing precision and recall.',
      formulaLatex: String.raw`F1_i = \max_{g \in G_i} \frac{2\,P(\hat{y}_i,g)\,R(\hat{y}_i,g)}{P(\hat{y}_i,g)+R(\hat{y}_i,g)},\quad F1=\frac{1}{N}\sum_{i=1}^{N}F1_i`,
      interpretation: 'Higher is better. Sensitive to missing key tokens and extra unsupported tokens.',
      variables: ['\(\hat{y}_i\): generated answer (`llm_ans`)', '\(G_i\): reference set parsed from `answer`/`answers`', '\(N\): sample count'],
      projectExample: ['Sample llm_ans: "...brother was Thietgaud." while the reference includes "...sister of Thietgaud...".', 'Compute token-level precision/recall per reference, keep the best reference score for each sample, then average across all samples.'],
    },
    rouge_l: {
      title: 'ROUGE-L (F-measure)',
      blurb: 'Measures longest common subsequence consistency between prediction and reference.',
      formulaLatex: String.raw`ROUGE\text{-}L_i = \max_{g \in G_i} \frac{(1+\beta^2)P_{LCS}(\hat{y}_i,g)R_{LCS}(\hat{y}_i,g)}{R_{LCS}(\hat{y}_i,g)+\beta^2P_{LCS}(\hat{y}_i,g)},\quad ROUGE\text{-}L=\frac{1}{N}\sum_i ROUGE\text{-}L_i`,
      interpretation: 'Higher is better when answer sequence structure is close to reference wording.',
      variables: ['\(P_{LCS}\), \(R_{LCS}\): based on LCS length', '\(\beta\): default balance factor in rouge implementation'],
      projectExample: ['This score increases when llm_ans preserves the key sequence structure of the reference.'],
    },
    bleu_1: {
      title: 'BLEU-1',
      blurb: 'Unigram precision with brevity penalty.',
      formulaLatex: String.raw`BLEU\text{-}1 = BP \cdot \exp(\log p_1)`,
      interpretation: 'Higher favors lexical correctness at word level.',
      variables: ['\(p_1\): modified 1-gram precision', '\(BP\): brevity penalty'],
      projectExample: ['Useful for checking whether key reference terms are present in the generated answer.'],
    },
    bleu_2: {
      title: 'BLEU-2',
      blurb: 'Up to bigram precision with smoothing.',
      formulaLatex: String.raw`BLEU\text{-}2 = BP \cdot \exp\left(\frac{1}{2}(\log p_1 + \log p_2)\right)`,
      interpretation: 'Higher rewards short phrase-level correctness.',
      variables: ['\(p_2\): modified 2-gram precision'],
      projectExample: ['BLEU-2 improves when short phrases (e.g., "bishop of Trier") are matched exactly.'],
    },
    bleu_3: {
      title: 'BLEU-3',
      blurb: 'Up to trigram precision with smoothing.',
      formulaLatex: String.raw`BLEU\text{-}3 = BP \cdot \exp\left(\frac{1}{3}(\log p_1 + \log p_2 + \log p_3)\right)`,
      interpretation: 'Higher indicates better multi-token phrase fluency and alignment.',
      variables: ['\(p_3\): modified 3-gram precision'],
      projectExample: ['Better for measuring local phrase-level alignment beyond unigram overlap.'],
    },
    bleu_4: {
      title: 'BLEU-4',
      blurb: 'Up to 4-gram precision; strict lexical-sequence match.',
      formulaLatex: String.raw`BLEU\text{-}4 = BP \cdot \exp\left(\frac{1}{4}\sum_{n=1}^{4}\log p_n\right)`,
      interpretation: 'Higher is harder to achieve; very sensitive to exact phrasing.',
      variables: ['\(p_n\): modified n-gram precision for \(n=1..4\)'],
      projectExample: ['Helpful for comparing whether different chunking strategies produce phrasing closer to reference wording.'],
    },
    bert_score_f1: {
      title: 'BERTScore F1',
      blurb: 'Semantic similarity using contextual embeddings (beyond exact string overlap).',
      formulaLatex: String.raw`BERTScore\text{-}F1 = \frac{2\,P_{bert}\,R_{bert}}{P_{bert}+R_{bert}}`,
      interpretation: 'Higher is better for semantic faithfulness even with paraphrases.',
      variables: ['\(P_{bert},R_{bert}\): token embedding alignment precision/recall'],
      projectExample: ['When llm_ans is a paraphrase of the reference, BERTScore is often higher than BLEU/ROUGE.'],
    },
    sample_count: {
      title: 'Sample Count',
      blurb: 'Number of evaluated records in current run.',
      formulaLatex: String.raw`N = |\mathcal{D}|`,
      interpretation: 'Not a quality metric; used to judge statistical stability of summary scores.',
      variables: ['\(\mathcal{D}\): evaluation dataset after parsing'],
      projectExample: ['Larger sample size improves stability of mean scores; with small N, inspect sample-level details together.'],
    },
  };

  const ragasMetricInfo: Record<string, MetricCardDoc> = {
    ragas_score: {
      title: 'Project RAGAS Score (Local Aggregation)',
      blurb: 'Project-specific aggregate implemented in eval_ragas.py, not an official single metric in Ragas.',
      formulaLatex: String.raw`s_i=\frac{1}{|M_i^+|}\sum_{m\in M_i^+}m_i,\;M_i^+=\{m\in\{faithfulness,answer\_relevancy,context\_recall,context\_precision,context\_entity\_recall\}\mid m_i>0\};\quad ragas\_score_{mean}=\frac{1}{|S^+|}\sum_{i\in S^+}s_i`,
      interpretation: 'Higher is better. Noise sensitivity metrics are excluded because they are lower-is-better.',
      variables: ['\(s_i\): sample-level local aggregate', '\(S^+\): samples with valid positive aggregate'],
      projectExample: ['For each sample, aggregate the five positive-direction metrics first, then report dataset-level mean/min/max.'],
    },
    faithfulness: {
      title: 'Faithfulness (Ragas)',
      blurb: 'Checks whether response claims are supported by retrieved contexts.',
      formulaLatex: String.raw`Faithfulness = \frac{\#\text{supported claims in response}}{\#\text{claims in response}}`,
      interpretation: 'Higher is better; low values indicate hallucination risk.',
      variables: ['Claims are decomposed from the generated response text in `answer`', 'Support is judged by attribution to evidence in `contexts`'],
      projectExample: ['If llm_ans includes an unsupported date/entity relation not found in retrieved context, this score drops.'],
    },
    answer_relevancy: {
      title: 'Answer Relevancy (Ragas)',
      blurb: 'Measures whether response addresses the original question intent.',
      formulaLatex: String.raw`Answer\;Relevancy=\frac{1}{N}\sum_{j=1}^{N}\cos(E_{g_j},E_o)=\frac{1}{N}\sum_{j=1}^{N}\frac{E_{g_j}\cdot E_o}{\|E_{g_j}\|\,\|E_o\|}`,
      interpretation: 'Higher is better for question-answer alignment, independent from factuality.',
      variables: ['\(E_o\): embedding of user question', '\(E_{g_j}\): embedding of synthetic question reversed from response'],
      projectExample: ['If the response addresses only part of the intent (e.g., place but not time), the score decreases.'],
    },
    context_recall: {
      title: 'Context Recall (Ragas)',
      blurb: 'How much of reference-answer claims are covered by retrieved context.',
      formulaLatex: String.raw`Context\;Recall=\frac{\#\text{reference claims supported by retrieved context}}{\#\text{claims in reference}}`,
      interpretation: 'Higher is better; reflects retriever completeness.',
      variables: ['In this project, `ground_truth` is normalized from `answer`/`answers` before scoring'],
      projectExample: ['If `gold_reference` is incomplete, retrieval coverage decreases and recall is directly affected.'],
    },
    context_precision: {
      title: 'Context Precision@K (Ragas)',
      blurb: 'Whether relevant chunks appear early in retrieved list.',
      formulaLatex: String.raw`Context\;Precision@K=\frac{\sum_{k=1}^{K}(Precision@k\cdot v_k)}{\sum_{k=1}^{K} v_k},\quad Precision@k=\frac{TP@k}{TP@k+FP@k}`,
      interpretation: 'Higher is better; rewards ranking quality, penalizes early noise.',
      variables: ['\(v_k\in\{0,1\}\): relevance at rank \(k\)', '\(K\): number of retrieved contexts'],
      projectExample: ['This score is higher when the top-ranked `rag_retrieval` chunks already contain key evidence.'],
    },
    context_entity_recall: {
      title: 'Context Entity Recall (Ragas)',
      blurb: 'Entity coverage from reference by retrieved contexts.',
      formulaLatex: String.raw`Context\;Entity\;Recall=\frac{|RCE\cap RE|}{|RE|}`,
      interpretation: 'Higher is better in entity-centric QA tasks.',
      variables: ['\(RE\): entities from reference', '\(RCE\): entities from retrieved contexts'],
      projectExample: ['For entity-heavy questions (people/locations/years), coverage of those entities in retrieval increases the score.'],
    },
    noise_sensitivity_relevant: {
      title: 'Noise Sensitivity (Relevant mode)',
      blurb: 'Error rate in response even when using relevant retrieved content.',
      formulaLatex: String.raw`NoiseSensitivity_{relevant}=\frac{\#\text{incorrect claims in response}}{\#\text{claims in response}}`,
      interpretation: 'Lower is better; reflects generation robustness under partially noisy evidence.',
      variables: ['Incorrectness is judged against `ground_truth` and evidence-attributable contexts'],
      projectExample: ['Even with relevant retrieval, this score rises if the model still fabricates unsupported details.'],
    },
    noise_sensitivity_irrelevant: {
      title: 'Noise Sensitivity (Irrelevant mode)',
      blurb: 'How often irrelevant contexts mislead generation into incorrect claims.',
      formulaLatex: String.raw`NoiseSensitivity_{irrelevant}=\frac{\#\text{incorrect claims triggered by irrelevant context}}{\#\text{claims in response}}`,
      interpretation: 'Lower is better; high score indicates vulnerability to retrieval distraction.',
      variables: ['irrelevant mode follows Ragas NoiseSensitivity(mode="irrelevant")'],
      projectExample: ['This value increases when irrelevant context distracts generation into incorrect claims.'],
    },
  };

  // Traditional Eval - Direct Input
  const [testDataJson, setTestDataJson] = useState('');
  const [enableBertScore, setEnableBertScore] = useState(false);
  const [traditionalResult, setTraditionalResult] = useState<any>(null);

  // Traditional Eval - File Input
  const [tempTraditionalPath, setTempTraditionalPath] = useState('');
  const [traditionalOutputPath, setTraditionalOutputPath] = useState('');
  const [traditionalFilePaths, setTraditionalFilePaths] = useState<string[]>([]);
  const traditionalFileRef = useRef<HTMLInputElement>(null);

  // RAGAS Eval - Direct Input
  const [ragasDataJson, setRagasDataJson] = useState('');
  const [ragasResult, setRagasResult] = useState<any>(null);

  // RAGAS Eval - File Input
  const [tempRagasPath, setTempRagasPath] = useState('');
  const [ragasOutputPath, setRagasOutputPath] = useState('');
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
      const testData = JSON.parse(testDataJson);

      const data: any = {
        test: testData,
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
      const data: any = {
        input_path: traditionalFilePaths[0],
        enable_bert_score: enableBertScore,
      };
      if (traditionalOutputPath.trim()) {
        data.output_path = traditionalOutputPath.trim();
      }

      const response = await api.traditionalEvalFile(data);
      if (response.success) {
        setTraditionalResult(response.data);
        toast.success(
          traditionalOutputPath.trim()
            ? `Evaluation completed. Summary saved to ${traditionalOutputPath.trim()}`
            : 'Evaluation completed'
        );
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
        input_path: ragasFilePaths[0],
        output_path: ragasOutputPath.trim() || undefined,
        vllm_api_base: vllmApiBase,
        vllm_model_name: vllmModelName,
        embedding_model_path: embeddingModelPath,
      });

      if (response.success) {
        setRagasResult(response.data);
        toast.success(
          ragasOutputPath.trim()
            ? `RAGAS evaluation completed. Summary saved to ${ragasOutputPath.trim()}`
            : 'RAGAS evaluation completed'
        );
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
      const ragasData = JSON.parse(ragasDataJson);

      const data: any = {
        test: ragasData,
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
          <h1 className="text-3xl font-bold mb-2">End-to-End Eval</h1>
          <p className="text-slate-600">Evaluate RAG system quality using traditional metrics and RAGAS.</p>
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
                        onKeyDown={(e) => fillPlaceholderOnTab(e, testDataJson, traditionalPlaceholder, setTestDataJson)}
                        placeholder={traditionalPlaceholder}
                        className="min-h-[280px] font-mono text-xs"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        💡 Press Tab on an empty editor to insert the example. Each item should include: _id, input, llm_ans, answer, rag_retrieval, gold_reference
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
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempTraditionalPath, e.currentTarget.placeholder, setTempTraditionalPath)}
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

                    <div>
                      <Label>Output Summary JSON Path</Label>
                      <Input
                        value={traditionalOutputPath}
                        onChange={(e) => setTraditionalOutputPath(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, traditionalOutputPath, e.currentTarget.placeholder, setTraditionalOutputPath)}
                        placeholder="/path/to/traditional_eval_summary.json"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Optional. If provided, the backend will write the summary metrics to this JSON file.
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
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h2 className="font-bold">Evaluation Results</h2>
                    <p className="text-xs text-slate-500 mt-1">Click a metric card to view a short explanation and formula.</p>
                  </div>
                  <Dialog open={!!selectedTraditionalMetric} onOpenChange={(open) => !open && setSelectedTraditionalMetric(null)}>
                    <DialogContent className="sm:max-w-[560px]">
                      {selectedTraditionalMetric && traditionalMetricInfo[selectedTraditionalMetric] && (
                        <>
                          <DialogHeader>
                            <DialogTitle>{traditionalMetricInfo[selectedTraditionalMetric].title}</DialogTitle>
                            <DialogDescription>
                              {traditionalMetricInfo[selectedTraditionalMetric].blurb}
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4">
                            <div className="rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 via-cyan-50 to-white p-4 shadow-sm">
                              <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-700">
                                <Sigma className="h-3.5 w-3.5" /> Formula (LaTeX)
                              </div>
                              <div className="rounded-lg bg-white/80 p-3 text-blue-950">
                                <BlockMath math={traditionalMetricInfo[selectedTraditionalMetric].formulaLatex} />
                              </div>
                            </div>
                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                              <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500 mb-2">Interpretation</div>
                              <p className="text-sm text-slate-700">{traditionalMetricInfo[selectedTraditionalMetric].interpretation}</p>
                            </div>
                            <div className="rounded-xl border border-slate-200 bg-white p-4">
                              <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Variables</div>
                              <ul className="space-y-1 text-sm text-slate-700 list-disc pl-5">
                                {traditionalMetricInfo[selectedTraditionalMetric].variables.map((item, idx) => (
                                  <li key={idx}>{item}</li>
                                ))}
                              </ul>
                            </div>
                            <div className="rounded-xl border border-blue-100 bg-blue-50/60 p-4">
                              <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-blue-700"><FlaskConical className="h-3.5 w-3.5" />Project-aligned example</div>
                              <ul className="space-y-1 text-sm text-blue-900 list-disc pl-5">
                                {traditionalMetricInfo[selectedTraditionalMetric].projectExample.map((item, idx) => (
                                  <li key={idx}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </>
                      )}
                    </DialogContent>
                  </Dialog>
                </div>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : traditionalResult ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-6">
                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(traditionalResult)
                          .filter(([key, value]) => typeof value === 'number' && !['bleu_2', 'bleu_3', 'bleu_4'].includes(key))
                          .map(([key, value]) => {
                            if (key === 'bleu_1') {
                              const bleuKeys = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4'];
                              const bleuValues = bleuKeys.map((k) => Number(traditionalResult[k] ?? 0));
                              const bleuMax = Math.max(...bleuValues, 1e-6);
                              const bleuMean = bleuValues.reduce((a, b) => a + b, 0) / bleuValues.length;
                              return (
                                <div key="bleu_family" className="col-span-2 rounded-xl border border-blue-200 bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50 p-4 text-left shadow-sm">
                                  <button
                                    type="button"
                                    className="w-full flex items-start justify-between gap-3"
                                    onClick={() => setBleuExpanded((v) => !v)}
                                  >
                                    <div className="flex-1">
                                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Lexical Overlap Panel</div>
                                      <div className="mt-1 text-sm text-slate-700">BLEU Family (n = 1..4)</div>
                                      <div className="mt-1 flex items-end gap-4">
                                        <div className="text-2xl font-bold text-blue-900">{Number(traditionalResult.bleu_4 ?? traditionalResult.bleu_1).toFixed(4)}</div>
                                        <div className="text-xs text-slate-600 pb-1">BLEU-4 (strict) · Mean {bleuMean.toFixed(4)}</div>
                                      </div>
                                      <p className="mt-1 text-xs text-slate-600">Paper-style compact block. Expand to inspect each n-gram order and open formulas.</p>
                                    </div>
                                    {bleuExpanded ? <ChevronUp className="w-4 h-4 text-blue-700 mt-1" /> : <ChevronDown className="w-4 h-4 text-blue-700 mt-1" />}
                                  </button>

                                  <div className="mt-3 rounded-lg border border-blue-100 bg-white/70 p-3">
                                    <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-500">N-gram profile (normalized bars)</div>
                                    <div className="grid grid-cols-4 gap-2">
                                      {bleuKeys.map((bleuKey, idx) => {
                                        const v = Number(traditionalResult[bleuKey] ?? 0);
                                        const h = Math.max(8, Math.round((v / bleuMax) * 42));
                                        return (
                                          <button
                                            key={`bleu_bar_${bleuKey}`}
                                            type="button"
                                            onClick={() => setSelectedTraditionalMetric(bleuKey)}
                                            className="rounded-md border border-blue-100 bg-white p-2 text-center hover:border-blue-300"
                                          >
                                            <div className="mx-auto mb-2 w-6 rounded-sm bg-blue-500/80" style={{ height: `${h}px` }} />
                                            <div className="text-[10px] uppercase text-slate-500">B{idx + 1}</div>
                                            <div className="text-xs font-semibold text-blue-900">{v.toFixed(4)}</div>
                                          </button>
                                        );
                                      })}
                                    </div>
                                  </div>

                                  {bleuExpanded && (
                                    <div className="mt-3 grid grid-cols-2 gap-2">
                                      {bleuKeys.map((bleuKey) => (
                                        typeof traditionalResult[bleuKey] === 'number' ? (
                                          <button
                                            key={bleuKey}
                                            type="button"
                                            onClick={() => setSelectedTraditionalMetric(bleuKey)}
                                            className="rounded-md border border-blue-200 bg-white/90 px-3 py-2 text-left hover:bg-white"
                                          >
                                            <div className="text-xs text-slate-600 uppercase">{bleuKey.replace('_', '-')}</div>
                                            <div className="text-sm font-semibold text-blue-900">{Number(traditionalResult[bleuKey]).toFixed(4)}</div>
                                          </button>
                                        ) : null
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            }

                            const metricInfo = traditionalMetricInfo[key];
                            return (
                              <button
                                key={key}
                                type="button"
                                onClick={() => metricInfo && setSelectedTraditionalMetric(key)}
                                className="p-4 text-left bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-blue-300 disabled:cursor-default disabled:hover:translate-y-0 disabled:hover:shadow-none"
                                disabled={!metricInfo}
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="text-sm text-slate-600 mb-1">{key.replace(/_/g, ' ').toUpperCase()}</div>
                                  {metricInfo && (
                                    <span className="rounded-full border border-blue-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-blue-700">Formula</span>
                                  )}
                                </div>
                                <div className="text-2xl font-bold text-blue-900">{key === 'sample_count' ? Math.round(value as number) : (value as number).toFixed(4)}</div>
                                {metricInfo && <p className="mt-2 text-xs leading-5 text-slate-600">{metricInfo.blurb}</p>}
                              </button>
                            );
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, vllmApiBase, e.currentTarget.placeholder, setVllmApiBase)}
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, vllmModelName, e.currentTarget.placeholder, setVllmModelName)}
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, embeddingModelPath, e.currentTarget.placeholder, setEmbeddingModelPath)}
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
                        onKeyDown={(e) => fillPlaceholderOnTab(e, ragasDataJson, ragasPlaceholder, setRagasDataJson)}
                        placeholder={ragasPlaceholder}
                        className="min-h-[360px] font-mono text-xs"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        💡 Press Tab on an empty editor to insert the example. Supports both formats and auto-converts on the backend.
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, vllmApiBase, e.currentTarget.placeholder, setVllmApiBase)}
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, vllmModelName, e.currentTarget.placeholder, setVllmModelName)}
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
                              onKeyDown={(e) => fillPlaceholderOnTab(e, embeddingModelPath, e.currentTarget.placeholder, setEmbeddingModelPath)}
                              placeholder="/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
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
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempRagasPath, e.currentTarget.placeholder, setTempRagasPath)}
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

                    <div>
                      <Label>Output Summary JSON Path</Label>
                      <Input
                        value={ragasOutputPath}
                        onChange={(e) => setRagasOutputPath(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, ragasOutputPath, e.currentTarget.placeholder, setRagasOutputPath)}
                        placeholder="/path/to/ragas_eval_summary.json"
                      />
                      <p className="text-xs text-slate-500 mt-1">
                        Optional. If provided, the backend will write the summary result to this JSON file.
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
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h2 className="font-bold">RAGAS Results</h2>
                    <p className="text-xs text-slate-500 mt-1">Click a summary card to view a short explanation and formula.</p>
                  </div>
                  <Dialog open={!!selectedRagasMetric} onOpenChange={(open) => !open && setSelectedRagasMetric(null)}>
                    <DialogContent className="sm:max-w-[560px]">
                      {selectedRagasMetric && ragasMetricInfo[selectedRagasMetric] && (
                        <>
                          <DialogHeader>
                            <DialogTitle>{ragasMetricInfo[selectedRagasMetric].title}</DialogTitle>
                            <DialogDescription>
                              {ragasMetricInfo[selectedRagasMetric].blurb}
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4">
                            <div className="rounded-2xl border border-purple-200 bg-gradient-to-br from-purple-50 via-fuchsia-50 to-white p-4 shadow-sm">
                              <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-purple-700">
                                <Sigma className="h-3.5 w-3.5" /> Formula (LaTeX)
                              </div>
                              <div className="rounded-lg bg-white/80 p-3 text-purple-950">
                                <BlockMath math={ragasMetricInfo[selectedRagasMetric].formulaLatex} />
                              </div>
                            </div>
                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                              <div className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500 mb-2">Interpretation</div>
                              <p className="text-sm text-slate-700">{ragasMetricInfo[selectedRagasMetric].interpretation}</p>
                            </div>
                            <div className="rounded-xl border border-slate-200 bg-white p-4">
                              <div className="mb-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">Variables</div>
                              <ul className="space-y-1 text-sm text-slate-700 list-disc pl-5">
                                {ragasMetricInfo[selectedRagasMetric].variables.map((item, idx) => (
                                  <li key={idx}>{item}</li>
                                ))}
                              </ul>
                            </div>
                            <div className="rounded-xl border border-purple-100 bg-purple-50/70 p-4">
                              <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-purple-700"><FlaskConical className="h-3.5 w-3.5" />Project-aligned example</div>
                              <ul className="space-y-1 text-sm text-purple-900 list-disc pl-5">
                                {ragasMetricInfo[selectedRagasMetric].projectExample.map((item, idx) => (
                                  <li key={idx}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          </div>
                        </>
                      )}
                    </DialogContent>
                  </Dialog>
                </div>
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
                              const metricInfo = ragasMetricInfo[key];
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
                                <button
                                  key={key}
                                  type="button"
                                  onClick={() => metricInfo && setSelectedRagasMetric(key)}
                                  className="p-3 text-left bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200 transition-all hover:-translate-y-0.5 hover:shadow-md hover:border-purple-300 disabled:cursor-default disabled:hover:translate-y-0 disabled:hover:shadow-none"
                                  disabled={!metricInfo}
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="text-xs text-slate-600 mb-1">
                                      {key.replace(/_/g, ' ').toUpperCase()}
                                    </div>
                                    {metricInfo && (
                                      <span className="rounded-full border border-purple-300 bg-white/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.16em] text-purple-700">
                                        Formula
                                      </span>
                                    )}
                                  </div>
                                  <div className="text-xl font-bold text-purple-900">
                                    {displayValue}
                                  </div>
                                  {metricInfo && (
                                    <p className="mt-2 text-xs leading-5 text-slate-600">
                                      {metricInfo.blurb}
                                    </p>
                                  )}
                                  {value && typeof value === 'object' && 'mean' in value && (
                                    <div className="text-xs text-slate-500 mt-1">
                                      min: {(value as any).min.toFixed(3)} | max: {(value as any).max.toFixed(3)}
                                    </div>
                                  )}
                                </button>
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
                                <TableHead>ID</TableHead>
                                <TableHead>Question</TableHead>
                                <TableHead>Metrics</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {ragasResult.samples.slice(0, 10).map((sample: any, idx: number) => (
                                <TableRow key={idx}>
                                  <TableCell className="font-mono text-xs">
                                    {sample.index || sample.id || `#${idx + 1}`}
                                  </TableCell>
                                  <TableCell className="max-w-xs truncate">
                                    {sample.question || sample.input || '-'}
                                  </TableCell>
                                  <TableCell>
                                    <div className="text-xs space-y-1">
                                      {Object.entries(sample)
                                        .filter(([key]) => 
                                          key !== 'question' && 
                                          key !== 'input' && 
                                          key !== 'id' &&
                                          key !== 'answer' && 
                                          key !== 'ground_truth' &&
                                          key !== 'contexts' &&
                                          key !== 'retrieval_list' &&
                                          key !== 'llm_ans' &&
                                          key !== 'answers' &&
                                          key !== 'metrics'
                                        )
                                        .map(([key, val]) => (
                                          <div key={key}>
                                            <span className="text-slate-500">{key}:</span> {typeof val === 'number' ? (val as number).toFixed(3) : '-'}
                                          </div>
                                        ))}                                   </div>
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

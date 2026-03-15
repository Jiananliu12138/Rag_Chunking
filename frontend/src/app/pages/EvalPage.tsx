import React, { useState } from 'react';
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
import { Loader2, BarChart3, FileText, Sparkles, Settings, Plus, X, FlaskConical, Sigma, ChevronDown, ChevronUp } from 'lucide-react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { api } from '../utils/api';
import { toast } from 'sonner';
import { PathPickerButton } from '../components/PathPickerButton';

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
    sources?: Array<{ label: string; url: string }>;
  };

  const traditionalDefaultSources = [
    { label: 'LongBench benchmark repository', url: 'https://github.com/THUDM/LongBench' },
    { label: 'BLEU original paper (Papineni et al., 2002)', url: 'https://aclanthology.org/P02-1040/' },
    { label: 'ROUGE package paper (Lin, 2004)', url: 'https://aclanthology.org/W04-1013/' },
    { label: 'BERTScore paper (Zhang et al., 2020)', url: 'https://openreview.net/forum?id=SkeHuCVFDr' },
    { label: 'Project implementation: eval_lite.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_lite.py' },
  ];

  const ragasDefaultSources = [
    { label: 'Ragas GitHub Repository', url: 'https://github.com/explodinggradients/ragas' },
    { label: 'Ragas metrics documentation', url: 'https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/' },
    { label: 'RAGAS framework paper', url: 'https://arxiv.org/abs/2309.15217' },
    { label: 'Project implementation: eval_ragas.py', url: 'file:///F:/thesis/Meta-Chunking/eval/LongBench/eval_ragas.py' },
  ];

  const traditionalMetricInfo: Record<string, MetricCardDoc> = {
    f1: {
      title: 'F1 Score (Token-level, Max-over-References)',
      blurb: 'Measures overlap between llm_ans and ground truth, balancing precision and recall.',
      formulaLatex: String.raw`F1_i = \max_{g \in G_i} \frac{2\,P(\hat{y}_i,g)\,R(\hat{y}_i,g)}{P(\hat{y}_i,g)+R(\hat{y}_i,g)},\quad F1=\frac{1}{N}\sum_{i=1}^{N}F1_i`,
      interpretation: 'Higher is better, F1 is the harmonic mean of Precision and Recall, used to measure the overlap between predicted answers and actual answer tokens. In QA tasks, if multiple correct answers exist, the one with the highest F1 score among the predicted answers is selected.',
      variables: [String.raw`\hat{y}_i:\ \text{generated answer }(\mathtt{llm\_ans})`, String.raw`G_i:\ \text{ground-truth reference set}`, String.raw`N:\ \text{sample count}`,String.raw`P(\hat{y}_i, g)=\frac{|\hat{y}_i \cap g|}{|\hat{y}_i|}`,String.raw`R(\hat{y}_i, g)=\frac{|\hat{y}_i \cap g|}{|g|}`],
      projectExample: [
        'Prediction (`llm_ans`): "Waldrada\'s brother was Thietgaud."; ground truth: "Waldrada was the sister of Thietgaud, bishop of Trier."',
        'After project normalization (lowercase, remove punctuation/articles, collapse spaces), prediction becomes `waldradas brother was thietgaud`, and ground truth becomes `waldrada was sister of thietgaud bishop of trier`.',
        'Token sets used by `qa_f1_score`: prediction = [`waldradas`, `brother`, `was`, `thietgaud`] (4 tokens); ground truth = [`waldrada`, `was`, `sister`, `of`, `thietgaud`, `bishop`, `of`, `trier`] (8 tokens).',
        'Common tokens are [`was`, `thietgaud`], so `num_same = 2`, `Precision = 2/4 = 0.5`, `Recall = 2/8 = 0.25`, and `F1 = 2PR/(P+R) = 2×0.5×0.25/(0.5+0.25) ≈ 0.3333`.',
        'If multiple ground-truth references exist, the evaluator computes this score against each reference, keeps the maximum F1 for the sample, and then averages over all samples.',
      ],
    },
    rouge_l: {
      title: 'ROUGE-L (F-measure)',
      blurb: 'Measures longest common subsequence consistency between prediction and reference.',
      formulaLatex: String.raw`ROUGE\text{-}L_i = \max_{g \in G_i} \frac{(1+\beta^2)P_{LCS}(\hat{y}_i,g)R_{LCS}(\hat{y}_i,g)}{R_{LCS}(\hat{y}_i,g)+\beta^2P_{LCS}(\hat{y}_i,g)},\quad ROUGE\text{-}L=\frac{1}{N}\sum_i ROUGE\text{-}L_i`,
      interpretation: 'Higher is better when the prediction preserves the reference token order. The core quantity is \(LCS(\hat{y}, g)\), the longest common subsequence between prediction and ground truth: tokens do not need to be contiguous, but their relative order must stay consistent. ROUGE-L then converts this LCS length into precision and recall, and combines them with an F-style score.',
      variables: [
        String.raw`LCS(\hat{y}, g):\ \text{longest common subsequence between prediction and ground truth}`,
        String.raw`P_{LCS}=\frac{LCS(\hat{y}, g)}{|\hat{y}|}:\ \text{fraction of prediction tokens covered by the LCS}`,
        String.raw`R_{LCS}=\frac{LCS(\hat{y}, g)}{|g|}:\ \text{fraction of ground-truth tokens covered by the LCS}`,
        String.raw`\beta:\ \text{recall-weighting factor; a larger }\beta\text{ gives more weight to recall}`,
      ],
      projectExample: [
        'Example prediction: `Thietgaud was bishop of Trier`; ground truth: `Thietgaud bishop Trier`.',
        'Tokens: prediction = 5 [`Thietgaud`, `was`, `bishop`, `of`, `Trier`], ground truth = 3 [`Thietgaud`, `bishop`, `Trier`]. The LCS is [`Thietgaud`, `bishop`, `Trier`], so `LCS = 3`.',
        'Therefore `P_{LCS} = 3/5 = 0.6` and `R_{LCS} = 3/3 = 1.0`.',
        'With `\beta = 1.2`, `ROUGE-L = ((1+1.2^2) \times 0.6 \times 1.0) / (1.0 + 1.2^2 \times 0.6) \approx 0.78`.',
        'This example shows why ROUGE-L rewards correct ordering and broad coverage: even though `was` and `of` are extra tokens, the ordered subsequence still fully covers the ground truth, so recall stays perfect while precision drops slightly.',
      ],
    },
    bleu_1: {
      title: 'BLEU-1',
      blurb: 'Unigram precision with brevity penalty.',
      formulaLatex: String.raw`BLEU\text{-}1 = BP \cdot \exp(\log p_1)`,
      interpretation: 'BLEU-1 equals unigram precision multiplied by the brevity penalty (BP), and is used to measure word-level similarity between the generated text and the ground-truth reference. BLEU also discourages models from outputting answers that are too short, so the brevity penalty reduces the score when the prediction is shorter than the reference.',
      variables: [
        String.raw`p_1=\frac{\#\text{matched unigrams}}{\#\text{total unigrams in prediction}}:\ \text{fraction of prediction words that also appear in the reference}`,
        String.raw`BP:\ \text{penalizes overly short predictions so BLEU does not reward incomplete answers}`,
      ],
      projectExample: ['Useful for checking whether key reference terms are present in the generated answer.'],
    },
    bleu_2: {
      title: 'BLEU-2',
      blurb: 'Up to bigram precision with smoothing.',
      formulaLatex: String.raw`BLEU\text{-}2 = BP \cdot \exp\left(\frac{1}{2}(\log p_1 + \log p_2)\right)`,
      interpretation: 'BLEU-2 extends BLEU-1 by requiring not only correct individual words but also correct adjacent 2-word phrases. It combines unigram precision and bigram precision with the brevity penalty, so it better reflects short phrase-level similarity between the generated text and the ground-truth reference.',
      variables: [
        String.raw`p_2=\frac{\#\text{matched bigrams}}{\#\text{total bigrams in prediction}}:\ \text{fraction of predicted 2-word phrases that appear in the reference}`,
        String.raw`BP:\ \text{brevity penalty shared by all BLEU variants}`,
      ],
      projectExample: [
        'BLEU-2 requires contiguous matching: it counts continuous 2-gram matches, unlike ROUGE-L which only requires order consistency.',
        'Prediction: `Thietgaud was bishop of Trier`; ground truth: `Thietgaud bishop of Trier`.',
        'Matched bigrams include `bishop of` and `of Trier`, but `Thietgaud was` does not match `Thietgaud bishop`, so BLEU-2 is lower than BLEU-1.',
      ],
    },
    bleu_3: {
      title: 'BLEU-3',
      blurb: 'Up to trigram precision with smoothing.',
      formulaLatex: String.raw`BLEU\text{-}3 = BP \cdot \exp\left(\frac{1}{3}(\log p_1 + \log p_2 + \log p_3)\right)`,
      interpretation: 'BLEU-3 is stricter than BLEU-2 because it also checks whether 3-word phrases align between prediction and reference. A high BLEU-3 score suggests that the generated answer preserves not just vocabulary, but also local phrase structure and fluency at the trigram level.',
      variables: [
        String.raw`p_3=\frac{\#\text{matched trigrams}}{\#\text{total trigrams in prediction}}:\ \text{fraction of predicted 3-word phrases that appear in the reference}`,
        String.raw`BP:\ \text{brevity penalty shared by all BLEU variants}`,
      ],
      projectExample: [
        'BLEU-3 requires contiguous matching: it counts continuous 3-gram matches, unlike ROUGE-L which only requires order consistency.',
        'Prediction: `Thietgaud was bishop of Trier`; ground truth: `Thietgaud bishop of Trier`.',
        'The trigram `bishop of Trier` matches, but `Thietgaud was bishop` does not, so BLEU-3 rewards partial phrase alignment while remaining stricter than BLEU-2.',
      ],
    },
    bleu_4: {
      title: 'BLEU-4',
      blurb: 'Up to 4-gram precision; strict lexical-sequence match.',
      formulaLatex: String.raw`BLEU\text{-}4 = BP \cdot \exp\left(\frac{1}{4}\sum_{n=1}^{4}\log p_n\right)`,
      interpretation: 'BLEU-4 combines unigram through 4-gram precision and is much more sensitive to exact wording and phrase order than BLEU-1. It is commonly used as a stricter summary score for lexical-sequence matching: if the generated answer uses the right words but in a different phrasing, BLEU-4 can still be low.',
      variables: [
        String.raw`p_n=\frac{\#\text{matched }n\text{-grams}}{\#\text{total }n\text{-grams in prediction}}:\ \text{phrase-match rate at order }n`,
        String.raw`BP:\ \text{brevity penalty shared by all BLEU variants}`,
      ],
      projectExample: [
        'BLEU-4 requires contiguous matching: it counts continuous 4-gram matches, unlike ROUGE-L which only requires order consistency.',
        'Prediction: `Thietgaud was bishop of Trier`; ground truth: `Thietgaud bishop of Trier`.',
        'Even when several words overlap, the 4-gram structure differs, so BLEU-4 drops quickly. This is why BLEU-4 is useful when you want to distinguish loose lexical overlap from near-exact phrasing.',
      ],
    },
    bert_score_f1: {
      title: 'BERTScore F1',
      blurb: 'Semantic similarity using contextual embeddings (beyond exact string overlap).',
      formulaLatex: String.raw`BERTScore\text{-}F1 = \frac{2\,P_{bert}\,R_{bert}}{P_{bert}+R_{bert}}`,
      interpretation: 'This formula is the F1 version of BERTScore, used to evaluate semantic similarity between the generated text and the ground-truth reference. It looks similar to the traditional F1 formula, but its Precision and Recall are computed from BERT embedding-based semantic similarity rather than simple token overlap. As a result, BERTScore can stay high even when the prediction is a paraphrase of the reference.',
      variables: [
        String.raw`P_{bert}:\ \text{average of the maximum semantic similarity from each prediction token to the reference tokens}`,
        String.raw`P_{bert}=\frac{1}{|\hat{y}|}\sum_{i\in\hat{y}}\max_{j\in g}\mathrm{sim}(e_i,e_j)`,
        String.raw`R_{bert}:\ \text{average of the maximum semantic similarity from each reference token to the prediction tokens}`,
        String.raw`R_{bert}=\frac{1}{|g|}\sum_{j\in g}\max_{i\in\hat{y}}\mathrm{sim}(e_j,e_i)`,
        String.raw`e_i:\ \text{embedding of the }i\text{-th prediction token}`,
        String.raw`e_j:\ \text{embedding of the }j\text{-th reference token}`,
        String.raw`\mathrm{sim}(e_i,e_j):\ \text{cosine similarity between two token embeddings}`,
      ],
      projectExample: ['When llm_ans is a paraphrase of the reference, BERTScore is often higher than BLEU/ROUGE.'],
    },
    sample_count: {
      title: 'Sample Count',
      blurb: 'Number of evaluated records in current run.',
      formulaLatex: String.raw`N = |\mathcal{D}|`,
      interpretation: 'Not a quality metric; used to judge statistical stability of summary scores.',
      variables: [String.raw`\mathcal{D}:\ \text{evaluation dataset after parsing}`],
      projectExample: ['Larger sample size improves stability of mean scores; with small N, inspect sample-level details together.'],
    },
  };

  const ragasMetricInfo: Record<string, MetricCardDoc> = {
    ragas_score: {
      title: 'Project RAGAS Score (Local Aggregation)',
      blurb: 'Project-specific aggregate implemented in eval_ragas.py, not an official single metric in Ragas.',
      formulaLatex: String.raw`s_i=\frac{1}{|M_i^+|}\sum_{m\in M_i^+}m_i,\;M_i^+=\{m\in\{faithfulness,answer\_relevancy,context\_recall,context\_precision,context\_entity\_recall\}\mid m_i>0\};\quad ragas\_score_{mean}=\frac{1}{|S^+|}\sum_{i\in S^+}s_i`,
      interpretation: 'Higher is better. Noise sensitivity metrics are excluded because they are lower-is-better.',
      variables: [String.raw`s_i:\ \text{sample-level local aggregate}`, String.raw`S^{+}:\ \text{samples with valid positive aggregate}`],
      projectExample: ['For each sample, aggregate the five positive-direction metrics first, then report dataset-level mean/min/max.'],
    },
    faithfulness: {
      title: 'Faithfulness (Ragas)',
      blurb: 'Measures whether the factual claims in the generated response are supported by the retrieved context.',
      formulaLatex: String.raw`Faithfulness = \frac{C_{\text{supported}}}{C_{\text{total}}}`,
      interpretation: 'Higher is better; lower values indicate that the response contains claims not grounded in the retrieved context (potential hallucinations).',
      variables: [
        String.raw`C_{\text{total}}:\ \text{total number of factual claims extracted from the generated response}`,
        String.raw`C_{\text{supported}}:\ \text{number of claims that can be inferred or verified from the retrieved contexts}`
      ],
      projectExample: [
        'If the generated answer states a fact (e.g., a date, entity relation, or event) that cannot be verified from the retrieved context passages, that claim is marked unsupported, decreasing the faithfulness score.'
      ],
    },
    answer_relevancy: {
      title: 'Answer Relevancy (Ragas)',
      blurb: 'Measures how well the generated response aligns with the intent of the user question.',
      formulaLatex: String.raw`Answer\ Relevancy = \frac{1}{N}\sum_{j=1}^{N}\cos(E_{g_j}, E_o) = \frac{1}{N}\sum_{j=1}^{N}\frac{E_{g_j}\cdot E_o}{\|E_{g_j}\|\|E_o\|}`,
      interpretation: 'Higher is better; measures semantic alignment between the user question and the information implied by the generated response, independent of factual correctness.',
      variables: [
        String.raw`E_o:\ \text{embedding of the original user input (question)}`,
        String.raw`E_{g_j}:\ \text{embedding of the } j\text{-th synthetic question generated from the response}`,
        String.raw`N:\ \text{number of generated synthetic questions}`
      ],
      projectExample: [
        'If the response only partially addresses the question intent (e.g., mentions the location but omits the requested date), the reversed questions will diverge from the original query, lowering the relevancy score.'
      ],
    },
    context_recall: {
      title: 'Context Recall (Ragas)',
      blurb: 'Measures how much of the reference information is covered by the retrieved contexts.',
      formulaLatex: String.raw`Context\ Recall = \frac{C_{\text{supported}}}{C_{\text{total}}}`,
      interpretation: 'Higher is better; indicates that the retriever successfully retrieves most of the information required to support the reference answer.',
      variables: [
        String.raw`C_{\text{total}}:\ \text{total number of factual claims extracted from the reference answer}`,
        String.raw`C_{\text{supported}}:\ \text{number of reference claims that can be attributed to the retrieved contexts}`
      ],
      projectExample: [
        'If the reference answer contains multiple facts but the retrieved contexts only contain evidence for some of them, only those claims are counted as supported, lowering the context recall score.'
      ],
    },
    context_precision: {
      title: 'Context Precision (Ragas)',
      blurb: 'Measures whether relevant retrieved chunks are ranked ahead of irrelevant ones.',
      formulaLatex: String.raw`Context\ Precision@K = \frac{\sum_{k=1}^{K}\left(Precision@k \cdot v_k\right)}{\sum_{k=1}^{K} v_k}, \qquad Precision@k = \frac{TP@k}{TP@k + FP@k}`,
      interpretation: 'Higher is better; indicates that useful evidence appears early in the retrieved ranking, while irrelevant chunks near the top lower the score.',
      variables: [
        String.raw`K:\ \text{total number of retrieved contexts}`,
        String.raw`v_k \in \{0,1\}:\ \text{relevance indicator of the chunk at rank } k`,
        String.raw`Precision@k:\ \text{precision of the retrieved list up to rank } k`,
        String.raw`TP@k,\ FP@k:\ \text{number of relevant and irrelevant retrieved chunks up to rank } k`
      ],
      projectExample: [
        'If the first few retrieved chunks in `rag_retrieval` already contain the evidence needed to support the answer, context precision is high; if irrelevant chunks appear before the useful ones, the score decreases.'
      ],
    },
    context_entity_recall: {
      title: 'Context Entity Recall (Ragas)',
      blurb: 'Measures how many entities mentioned in the reference are covered by the retrieved contexts.',
      formulaLatex: String.raw`Context\ Entity\ Recall = \frac{|RCE \cap RE|}{|RE|}`,
      interpretation: 'Higher is better; especially useful in entity-centric tasks where retrieval should preserve key people, places, organizations, dates, or other named entities from the reference.',
      variables: [
        String.raw`RE:\ \text{set of entities extracted from the reference}`,
        String.raw`RCE:\ \text{set of entities extracted from the retrieved contexts}`
      ],
      projectExample: [
        'For entity-heavy questions involving people, locations, dates, or organizations, the score increases when the retrieved contexts cover the same key entities present in the reference.'
      ],
    },
    noise_sensitivity_relevant: {
      title: 'Noise Sensitivity (Relevant Mode, Ragas)',
      blurb: 'Measures how often the system produces incorrect claims even when using relevant retrieved contexts.',
      formulaLatex: String.raw`Noise\ Sensitivity_{relevant}=\frac{\#\text{incorrect claims in response}}{\#\text{claims in response}}`,
      interpretation: 'Lower is better; indicates that the model can correctly utilize relevant retrieved evidence without introducing unsupported or incorrect statements.',
      variables: [
        String.raw`\text{incorrect claims}:\ \text{statements in the response not supported by the reference}`,
        String.raw`\text{claims in response}:\ \text{total number of decomposed statements in the generated answer}`
      ],
      projectExample: [
        'Even when `rag_retrieval` returns relevant evidence, the score increases if the model fabricates unsupported facts or introduces extra incorrect statements.'
      ],
    },

    noise_sensitivity_irrelevant: {
      title: 'Noise Sensitivity (Irrelevant Mode, Ragas)',
      blurb: 'Measures how often irrelevant retrieved contexts mislead the model into generating incorrect claims.',
      formulaLatex: String.raw`Noise\ Sensitivity_{irrelevant}=\frac{\#\text{incorrect claims caused by irrelevant context}}{\#\text{claims in response}}`,
      interpretation: 'Lower is better; high values indicate that the model is easily distracted by irrelevant retrieved information.',
      variables: [
        String.raw`\text{incorrect claims}:\ \text{response statements inconsistent with the reference}`,
        String.raw`\text{claims in response}:\ \text{total number of statements in the generated answer}`
      ],
      projectExample: [
        'If irrelevant chunks in `rag_retrieval` cause the model to generate incorrect statements, this metric increases.'
      ],
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

  // RAGAS Eval - Direct Input
  const [ragasDataJson, setRagasDataJson] = useState('');
  const [ragasResult, setRagasResult] = useState<any>(null);

  // RAGAS Eval - File Input
  const [tempRagasPath, setTempRagasPath] = useState('');
  const [ragasOutputPath, setRagasOutputPath] = useState('');
  const [ragasFilePaths, setRagasFilePaths] = useState<string[]>([]);

  // RAGAS Configuration (shared between direct input and file input)
  const [ragasConfigOpen, setRagasConfigOpen] = useState(false);
  const [vllmApiBase, setVllmApiBase] = useState('http://localhost:8005/v1');
  const [vllmModelName, setVllmModelName] = useState('Qwen2.5-7B-Instruct');
  const [embeddingModelPath, setEmbeddingModelPath] = useState('/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5');

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

  const renderMetricInfoDialog = (
    selectedMetric: string | null,
    setSelectedMetric: (metric: string | null) => void,
    metricInfoMap: Record<string, MetricCardDoc>,
    defaultSources: Array<{ label: string; url: string }>,
    palette: {
      formulaCard: string;
      formulaText: string;
      exampleCard: string;
      exampleText: string;
      sourceChip: string;
      exampleBodyText: string;
    },
  ) => {
    const metric = selectedMetric ? metricInfoMap[selectedMetric] : null;
    const sources = metric?.sources ?? defaultSources;

    return (
      <Dialog open={!!selectedMetric} onOpenChange={(open) => !open && setSelectedMetric(null)}>
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
                  <p className="break-words text-sm text-slate-700">{metric.interpretation}</p>
                </div>
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

  const renderRagasConfigDialog = (embeddingPlaceholder = '/path/to/bge-large-en-v1.5') => (
    <Dialog open={ragasConfigOpen} onOpenChange={setRagasConfigOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
          <Settings className="w-4 h-4 text-slate-500 transition-colors hover:text-purple-600" />
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
            <p className="mt-1 text-xs text-slate-500">The base URL for the VLLM API endpoint</p>
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
            <p className="mt-1 text-xs text-slate-500">Path to the VLLM model for evaluation</p>
          </div>
          <div>
            <Label htmlFor="embedding-model-path-2">Embedding Model Path</Label>
            <Input
              id="embedding-model-path-2"
              value={embeddingModelPath}
              onChange={(e) => setEmbeddingModelPath(e.target.value)}
              onKeyDown={(e) => fillPlaceholderOnTab(e, embeddingModelPath, e.currentTarget.placeholder, setEmbeddingModelPath)}
              placeholder={embeddingPlaceholder}
              className="mt-1.5"
            />
            <p className="mt-1 text-xs text-slate-500">Path to the embedding model for semantic evaluation</p>
          </div>
        </div>
        <div className="flex justify-end">
          <Button onClick={() => setRagasConfigOpen(false)}>Done</Button>
        </div>
      </DialogContent>
    </Dialog>
  );

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
              <div className="lg:h-[860px]">
                <ScrollArea className="h-full lg:pr-2">
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
                        className="h-[280px] resize-none overflow-y-auto font-mono text-xs"
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
                          placeholder="/path/to/sample_results.json (enter server path and click +)"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempTraditionalPath.trim()) {
                              setTraditionalFilePaths([...traditionalFilePaths, tempTraditionalPath.trim()]);
                              setTempTraditionalPath('');
                            }
                          }}
                        />
                        <PathPickerButton
                          mode="file"
                          value={tempTraditionalPath}
                          allowedExtensions={['.json']}
                          title="Select Traditional Eval File"
                          description="This browser reads the filesystem on the machine running the backend service."
                          onSelect={(path) => setTraditionalFilePaths((prev) => [...prev, path])}
                        />
                        <Button
                          type="button"
                          onClick={() => {
                            if (tempTraditionalPath.trim()) {
                              setTraditionalFilePaths([...traditionalFilePaths, tempTraditionalPath.trim()]);
                              setTempTraditionalPath('');
                            }
                          }}
                          size="sm"
                          variant="outline"
                        >
                          <Plus className="w-4 h-4" />
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
                      <div className="flex gap-2">
                        <Input
                          value={traditionalOutputPath}
                          onChange={(e) => setTraditionalOutputPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, traditionalOutputPath, e.currentTarget.placeholder, setTraditionalOutputPath)}
                          placeholder="/path/to/traditional_eval_summary.json"
                        />
                        <PathPickerButton
                          mode="directory"
                          value={traditionalOutputPath}
                          title="Select Output Directory"
                          description="Pick the folder to write the summary file into, then edit the filename if needed."
                          onSelect={(selectedDirectory) => {
                            const currentName = traditionalOutputPath.split(/[\\/]/).filter(Boolean).at(-1);
                            const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                            setTraditionalOutputPath(
                              currentName && currentName.includes('.')
                                ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                                : selectedDirectory,
                            );
                          }}
                        />
                      </div>
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
                </ScrollArea>
              </div>

              {/* Results */}
              <Card className="flex h-[860px] flex-col p-6">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h2 className="font-bold">Evaluation Results</h2>
                    <p className="text-xs text-slate-500 mt-1">Click a metric card to view a short explanation and formula.</p>
                  </div>
                  {renderMetricInfoDialog(
                    selectedTraditionalMetric,
                    setSelectedTraditionalMetric,
                    traditionalMetricInfo,
                    traditionalDefaultSources,
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
                ) : traditionalResult ? (
                  <ScrollArea className="h-full">
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
                                      <p className="mt-1 text-xs text-slate-600">Expand to inspect each n-gram order and open formulas.</p>
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
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400">
                    <div className="text-center">
                    <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>Evaluation results will appear here</p>
                    </div>
                  </div>
                )}
                </div>
              </Card>
            </div>
          </TabsContent>

          {/* RAGAS Tab */}
          <TabsContent value="ragas">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Methods */}
              <div className="lg:h-[860px]">
                <ScrollArea className="h-full lg:pr-2">
                <div className="space-y-6">
                {/* Direct JSON Input */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-bold flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-purple-600" />
                      Direct JSON Input
                    </h2>
                    {renderRagasConfigDialog()}
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>RAGAS Data (Flexible JSON Format)</Label>
                      <Textarea
                        value={ragasDataJson}
                        onChange={(e) => setRagasDataJson(e.target.value)}
                        onKeyDown={(e) => fillPlaceholderOnTab(e, ragasDataJson, ragasPlaceholder, setRagasDataJson)}
                        placeholder={ragasPlaceholder}
                        className="h-[360px] resize-none overflow-y-auto font-mono text-xs"
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
                    {renderRagasConfigDialog('/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5')}
                  </div>
                  <div className="space-y-4">
                    <div>
                      <Label>Evaluation Data File Path</Label>
                      <div className="flex gap-2">
                        <Input
                          value={tempRagasPath}
                          onChange={(e) => setTempRagasPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, tempRagasPath, e.currentTarget.placeholder, setTempRagasPath)}
                          placeholder="/path/to/ragas_data.json (enter server path and click +)"
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && tempRagasPath.trim()) {
                              setRagasFilePaths([...ragasFilePaths, tempRagasPath.trim()]);
                              setTempRagasPath('');
                            }
                          }}
                        />
                        <PathPickerButton
                          mode="file"
                          value={tempRagasPath}
                          allowedExtensions={['.json']}
                          title="Select RAGAS Eval File"
                          description="This browser reads the filesystem on the machine running the backend service."
                          onSelect={(path) => setRagasFilePaths((prev) => [...prev, path])}
                        />
                        <Button
                          type="button"
                          onClick={() => {
                            if (tempRagasPath.trim()) {
                              setRagasFilePaths([...ragasFilePaths, tempRagasPath.trim()]);
                              setTempRagasPath('');
                            }
                          }}
                          size="sm"
                          variant="outline"
                        >
                          <Plus className="w-4 h-4" />
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
                      <div className="flex gap-2">
                        <Input
                          value={ragasOutputPath}
                          onChange={(e) => setRagasOutputPath(e.target.value)}
                          onKeyDown={(e) => fillPlaceholderOnTab(e, ragasOutputPath, e.currentTarget.placeholder, setRagasOutputPath)}
                          placeholder="/path/to/ragas_eval_summary.json"
                        />
                        <PathPickerButton
                          mode="directory"
                          value={ragasOutputPath}
                          title="Select Output Directory"
                          description="Pick the folder to write the summary file into, then edit the filename if needed."
                          onSelect={(selectedDirectory) => {
                            const currentName = ragasOutputPath.split(/[\\/]/).filter(Boolean).at(-1);
                            const separator = selectedDirectory.includes('\\') ? '\\' : '/';
                            setRagasOutputPath(
                              currentName && currentName.includes('.')
                                ? `${selectedDirectory}${selectedDirectory.endsWith('/') || selectedDirectory.endsWith('\\') ? '' : separator}${currentName}`
                                : selectedDirectory,
                            );
                          }}
                        />
                      </div>
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
                </ScrollArea>
              </div>

              {/* Results */}
              <Card className="flex h-[860px] flex-col p-6">
                <div className="flex items-start justify-between gap-4 mb-4">
                  <div>
                    <h2 className="font-bold">RAGAS Results</h2>
                    <p className="text-xs text-slate-500 mt-1">Click a summary card to view a short explanation and formula.</p>
                  </div>
                  {renderMetricInfoDialog(
                    selectedRagasMetric,
                    setSelectedRagasMetric,
                    ragasMetricInfo,
                    ragasDefaultSources,
                    {
                      formulaCard: 'rounded-2xl border border-purple-200 bg-gradient-to-br from-purple-50 via-fuchsia-50 to-white p-4 shadow-sm',
                      formulaText: 'text-purple-700',
                      exampleCard: 'rounded-xl border border-purple-100 bg-purple-50/70 p-4',
                      exampleText: 'text-purple-700',
                      exampleBodyText: 'text-purple-900',
                      sourceChip: 'border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100',
                    },
                  )}
                </div>
                <div className="min-h-0 flex-1">
                {loading ? (
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin" />
                  </div>
                ) : ragasResult ? (
                  <ScrollArea className="h-full">
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
                                  {value !== null && typeof value === 'object' && 'mean' in (value as Record<string, unknown>) && (
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
                  <div className="flex h-full min-h-[560px] items-center justify-center rounded-xl bg-slate-50 text-slate-400">
                    <div className="text-center">
                    <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>RAGAS results will appear here</p>
                    </div>
                  </div>
                )}
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

import { useState, useEffect, useMemo } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Textarea } from '../components/ui/textarea';
import { Switch } from '../components/ui/switch';
import { Slider } from '../components/ui/slider';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader2, Cpu, Activity, FileText, Network, Grid3x3 } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';
import ForceGraph2D from 'react-force-graph-2d';

export default function ComponentEvalPage() {
  const [loading, setLoading] = useState(false);

  // Chunk Quality - Direct Input
  const [qualityChunksJson, setQualityChunksJson] = useState('');
  const [enableSemanticSimilarity, setEnableSemanticSimilarity] = useState(true);
  const [enableBoundaryClarity, setEnableBoundaryClarity] = useState(true);
  const [qualityResult, setQualityResult] = useState<any>(null);

  // Chunk Quality - File Input
  const [qualityFilePath, setQualityFilePath] = useState('');

  // Chunk Stickiness - Direct Input
  const [stickinessChunksJson, setStickinessChunksJson] = useState('');
  const [threshold, setThreshold] = useState([0.8]);
  const [delta, setDelta] = useState([0.0]);
  const [stickinessResult, setStickinessResult] = useState<any>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState([0.8]); // For visualization

  // Chunk Stickiness - File Input
  const [stickinessFilePath, setStickinessFilePath] = useState('');

  const handleQualityEval = async () => {
    if (!qualityChunksJson.trim()) {
      toast.error('Please enter chunks in JSON format');
      return;
    }

    setLoading(true);
    try {
      // Parse JSON input
      const chunks = JSON.parse(qualityChunksJson);
      const data: any = { 
        chunks,
        enable_semantic_similarity: enableSemanticSimilarity,
        enable_boundary_clarity: enableBoundaryClarity,
      };

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
    if (!qualityFilePath) {
      toast.error('Please enter file path');
      return;
    }

    setLoading(true);
    try {
      const data: any = { 
        input_path: qualityFilePath,
        enable_semantic_similarity: enableSemanticSimilarity,
        enable_boundary_clarity: enableBoundaryClarity,
      };

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
      // Parse JSON input
      const chunks = JSON.parse(stickinessChunksJson);
      const response = await api.chunkStickiness({
        chunks,
        threshold: threshold[0],
        delta: delta[0],
      });

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

  const handleStickinessFileEval = async () => {
    if (!stickinessFilePath) {
      toast.error('Please enter file path');
      return;
    }

    setLoading(true);
    try {
      const response = await api.chunkStickinessFile({
        input_path: stickinessFilePath,
        threshold: threshold[0],
        delta: delta[0],
      });

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

  // Auto re-evaluate when parameters change (for stickiness)
  useEffect(() => {
    if (stickinessChunksJson && stickinessResult) {
      const debounce = setTimeout(() => {
        handleStickinessEval();
      }, 500);
      return () => clearTimeout(debounce);
    }
  }, [threshold, delta]);

  // Prepare chart data from quality result
  const qualityChartData = qualityResult?.details?.map((item: any, index: number) => ({
    name: `Chunk ${index}`,
    'Boundary Clarity': item.boundary_clarity,
    'Semantic Dissimilarity': item.semantic_dissimilarity,
  })) || [];

  // Prepare force graph data from stickiness result
  const forceGraphData = useMemo(() => {
    if (!stickinessResult?.graph_incomplete) return null;

    const nodes: any[] = [];
    const links: any[] = [];
    const graph = stickinessResult.graph_incomplete;

    // Create nodes
    const nodeIds = Object.keys(graph);
    nodeIds.forEach((id) => {
      nodes.push({
        id,
        name: `Chunk ${id}`,
      });
    });

    // Create links (only show high similarity edges)
    nodeIds.forEach((source) => {
      Object.entries(graph[source] || {}).forEach(([target, weight]: [string, any]) => {
        const similarity = 1 - weight; // Convert weight to similarity
        if (similarity > similarityThreshold[0]) {
          links.push({
            source,
            target,
            similarity,
            value: similarity, // For link width
          });
        }
      });
    });

    return { nodes, links };
  }, [stickinessResult, similarityThreshold]);

  // Prepare heatmap data from stickiness result
  const heatmapData = useMemo(() => {
    if (!stickinessResult?.graph_complete) return [];

    const graph = stickinessResult.graph_complete;
    const nodeIds = Object.keys(graph).sort((a, b) => parseInt(a) - parseInt(b));
    const data: any[] = [];

    nodeIds.forEach((i) => {
      nodeIds.forEach((j) => {
        const weight = graph[i]?.[j] ?? 1;
        const similarity = Math.max(0, 1 - weight);
        data.push({
          x: parseInt(i),
          y: parseInt(j),
          similarity,
        });
      });
    });

    return data;
  }, [stickinessResult]);

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

          {/* Chunk Quality Tab */}
          <TabsContent value="quality">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input */}
              <div className="space-y-6">
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Direct Input
                  </h2>
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
                        <Switch
                          checked={enableSemanticSimilarity}
                          onCheckedChange={setEnableSemanticSimilarity}
                        />
                      </div>
                      <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <div>
                          <Label className="text-sm">Boundary Clarity</Label>
                          <p className="text-xs text-slate-500">Perplexity ratio</p>
                        </div>
                        <Switch
                          checked={enableBoundaryClarity}
                          onCheckedChange={setEnableBoundaryClarity}
                        />
                      </div>
                    </div>

                    <Button
                      onClick={handleQualityEval}
                      disabled={loading}
                      className="w-full"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Evaluate Quality'
                      )}
                    </Button>
                  </div>
                </Card>

                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    File Input
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Results File Path</Label>
                      <Input
                        value={qualityFilePath}
                        onChange={(e) => setQualityFilePath(e.target.value)}
                        placeholder="/path/to/chunks.json"
                      />
                    </div>

                    <Button
                      onClick={handleQualityFileEval}
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
                <h2 className="font-bold mb-4">Quality Results</h2>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
                  </div>
                ) : qualityResult ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-6">
                      {/* Summary Stats */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
                          <div className="text-sm text-slate-600 mb-1">Average Boundary Clarity</div>
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

                      {/* Chart */}
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
                              <Line
                                type="monotone"
                                dataKey="Boundary Clarity"
                                stroke="#3b82f6"
                                strokeWidth={2}
                              />
                              <Line
                                type="monotone"
                                dataKey="Semantic Dissimilarity"
                                stroke="#a855f7"
                                strokeWidth={2}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      )}

                      {/* Details Table */}
                      {qualityResult.details && qualityResult.details.length > 0 && (
                        <div>
                          <h3 className="text-sm font-medium mb-3">Adjacent Chunk Details</h3>
                          <div className="space-y-2">
                            {qualityResult.details.map((detail: any, index: number) => (
                              <div
                                key={index}
                                className="p-3 bg-slate-50 rounded-lg border border-slate-200"
                              >
                                <div className="flex items-center justify-between text-sm">
                                  <span className="font-medium">
                                    Chunk {index} → {index + 1}
                                  </span>
                                  <div className="flex gap-4">
                                    <span className="text-blue-600">
                                      BC: {detail.boundary_clarity?.toFixed(4)}
                                    </span>
                                    <span className="text-purple-600">
                                      Dissim: {detail.semantic_dissimilarity?.toFixed(4)}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Raw JSON */}
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

          {/* Chunk Stickiness Tab */}
          <TabsContent value="stickiness">
            <div className="space-y-6">
              {/* Input Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <Cpu className="w-5 h-5" />
                    Direct Input
                  </h2>
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

                    <Button
                      onClick={handleStickinessEval}
                      disabled={loading}
                      className="w-full"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Evaluating...
                        </>
                      ) : (
                        'Evaluate Stickiness'
                      )}
                    </Button>
                  </div>
                </Card>

                <Card className="p-6">
                  <h2 className="font-bold mb-4 flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    File Input
                  </h2>
                  <div className="space-y-4">
                    <div>
                      <Label>Chunk Results File Path</Label>
                      <Input
                        value={stickinessFilePath}
                        onChange={(e) => setStickinessFilePath(e.target.value)}
                        placeholder="/path/to/chunks.json"
                      />
                    </div>

                    <Button
                      onClick={handleStickinessFileEval}
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
                      />
                      <p className="text-xs text-slate-500">
                        Controls which edges are considered "strong correlation"
                      </p>
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
                      />
                      <p className="text-xs text-slate-500">
                        Higher values favor adjacent chunks
                      </p>
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
                      />
                      <p className="text-xs text-slate-500">
                        Only show edges with similarity above this threshold
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Results Section */}
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
                      <p className="text-sm text-slate-600">
                        Lower values indicate stronger overall cohesion
                      </p>
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
                      <p className="text-sm text-slate-600">
                        Difference from complete shows threshold impact
                      </p>
                    </Card>
                  </div>

                  {/* Visualizations */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    {/* Force Graph */}
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Network className="w-5 h-5" />
                        <h3 className="font-bold">Chunk Dependency Graph</h3>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Force-directed layout showing semantic relationships
                      </p>
                      {forceGraphData && forceGraphData.nodes.length > 0 ? (
                        <div className="bg-slate-50 rounded-lg border border-slate-200" style={{ height: '500px' }}>
                          <ForceGraph2D
                            graphData={forceGraphData}
                            nodeLabel="name"
                            nodeAutoColorBy="id"
                            linkWidth={(link: any) => Math.max(1, link.value * 5)}
                            linkColor={() => 'rgba(59, 130, 246, 0.6)'}
                            nodeRelSize={6}
                            linkDirectionalParticles={2}
                            linkDirectionalParticleWidth={(link: any) => link.value * 4}
                          />
                        </div>
                      ) : (
                        <div className="bg-slate-50 rounded-lg border border-slate-200 h-[500px] flex items-center justify-center">
                          <div className="text-center text-slate-400">
                            <Network className="w-12 h-12 mx-auto mb-2 opacity-50" />
                            <p>No edges above similarity threshold</p>
                            <p className="text-xs">Try lowering the visualization threshold</p>
                          </div>
                        </div>
                      )}
                    </Card>

                    {/* Heatmap */}
                    <Card className="p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Grid3x3 className="w-5 h-5" />
                        <h3 className="font-bold">Similarity Heatmap</h3>
                      </div>
                      <p className="text-sm text-slate-600 mb-4">
                        Matrix view of pairwise chunk similarities
                      </p>
                      {heatmapData.length > 0 ? (
                        <ScrollArea className="h-[500px]">
                          <div className="grid gap-1" style={{
                            gridTemplateColumns: `repeat(${Math.sqrt(heatmapData.length)}, minmax(0, 1fr))`
                          }}>
                            {heatmapData.map((cell, idx) => (
                              <div
                                key={idx}
                                className="aspect-square rounded-sm border border-slate-200"
                                style={{
                                  backgroundColor: `rgba(59, 130, 246, ${cell.similarity})`,
                                }}
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

                  {/* Interpretation Guide */}
                  <Card className="p-6 bg-blue-50 border-blue-200">
                    <h3 className="font-bold mb-3 text-blue-900">How to Interpret</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <h4 className="font-medium text-blue-800 mb-2">Entropy Values</h4>
                        <ul className="space-y-1 text-blue-700">
                          <li>• Lower entropy = stronger semantic coherence</li>
                          <li>• Large difference = many weak connections filtered</li>
                          <li>• Similar values = most chunks are related</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-blue-800 mb-2">Visualizations</h4>
                        <ul className="space-y-1 text-blue-700">
                          <li>• Dense clusters = chunks that should stay together</li>
                          <li>• Isolated nodes = distinct semantic topics</li>
                          <li>• Diagonal patterns in heatmap = sequential coherence</li>
                        </ul>
                      </div>
                    </div>
                  </Card>

                  {/* Raw JSON */}
                  <Card className="p-6">
                    <h3 className="font-bold mb-3">Complete Results</h3>
                    <ScrollArea className="h-[300px]">
                      <pre className="p-4 bg-slate-50 rounded-lg text-xs overflow-auto">
                        {JSON.stringify(stickinessResult, null, 2)}
                      </pre>
                    </ScrollArea>
                  </Card>
                </>
              ) : (
                <div className="text-center py-12 text-slate-400">
                  <Cpu className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Stickiness results will appear here</p>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

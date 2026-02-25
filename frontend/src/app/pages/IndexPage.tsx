import { useState, useEffect, useRef } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Badge } from '../components/ui/badge';
import { Switch } from '../components/ui/switch';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '../components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import { Loader2, Plus, Trash2, Database, Eye, RefreshCw, FileText, Filter, X, FolderOpen, AlertCircle } from 'lucide-react';
import { api } from '../utils/api';
import { toast } from 'sonner';
import { ScrollArea } from '../components/ui/scroll-area';

interface Collection {
  name: string;
  db_file: string;
  size_bytes: number;
}

export default function IndexPage() {
  const [collections, setCollections] = useState<Collection[]>([]);
  const [inspectData, setInspectData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [buildLoading, setBuildLoading] = useState(false);

  // Build form
  const [collectionName, setCollectionName] = useState('');
  const [docsPaths, setDocsPaths] = useState<string[]>([]);
  const [tempDocPath, setTempDocPath] = useState(''); // For adding files
  const [batchSize, setBatchSize] = useState(100);
  const [enableSparse, setEnableSparse] = useState(false);
  const [embedModelPath, setEmbedModelPath] = useState('');
  const [embedDim, setEmbedDim] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Add form
  const [addCollectionName, setAddCollectionName] = useState('');
  const [addDocsPaths, setAddDocsPaths] = useState<string[]>([]);
  const [tempAddDocPath, setTempAddDocPath] = useState(''); // For adding files
  const [addBatchSize, setAddBatchSize] = useState(100);
  const [addEmbedModelPath, setAddEmbedModelPath] = useState('');
  const addFileInputRef = useRef<HTMLInputElement>(null);

  // Delete documents dialog
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState('');
  const [documents, setDocuments] = useState<any[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [deleteFilterType, setDeleteFilterType] = useState<'filepath' | 'doc_id'>('filepath');
  const [selectedFilepath, setSelectedFilepath] = useState('');
  const [selectedDocId, setSelectedDocId] = useState('');
  const [availableFilepaths, setAvailableFilepaths] = useState<string[]>([]);
  const [availableDocIds, setAvailableDocIds] = useState<string[]>([]);

  useEffect(() => {
    loadCollections();
  }, []);

  const loadCollections = async () => {
    setLoading(true);
    try {
      const response = await api.listCollections();
      if (response.success) {
        setCollections(response.data.collections || []);
      } else {
        toast.error('Failed to load collections: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const loadInspectData = async () => {
    setLoading(true);
    try {
      const response = await api.inspectCollections();
      if (response.success) {
        setInspectData(response.data);
      } else {
        toast.error('Failed to inspect collections: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBuildIndex = async () => {
    if (!collectionName || !docsPaths.length) {
      toast.error('Please fill in collection name and documents path');
      return;
    }

    setBuildLoading(true);
    try {
      const requestData: any = {
        collection_name: collectionName,
        docs_paths: docsPaths,
        batch_size: batchSize,
        enable_sparse: enableSparse,
      };
      
      if (embedModelPath.trim()) {
        requestData.embed_model_path = embedModelPath.trim();
      }
      
      if (embedDim.trim()) {
        requestData.embed_dim = parseInt(embedDim);
      }

      const response = await api.buildIndex(requestData);

      if (response.success) {
        toast.success(
          `Index built! Indexed ${response.data.indexed_chunks}/${response.data.total_chunks} chunks in ${response.data.time_cost?.toFixed(2)}s`
        );
        setCollectionName('');
        setDocsPaths([]);
        setEmbedModelPath('');
        setEmbedDim('');
        loadCollections();
      } else {
        toast.error('Failed to build index: ' + response.message);
      }
    } finally {
      setBuildLoading(false);
    }
  };

  const handleAddIndex = async () => {
    if (!addCollectionName || !addDocsPaths.length) {
      toast.error('Please fill in collection name and documents path');
      return;
    }

    setBuildLoading(true);
    try {
      const requestData: any = {
        collection_name: addCollectionName,
        docs_paths: addDocsPaths,
        batch_size: addBatchSize,
      };
      
      if (addEmbedModelPath.trim()) {
        requestData.embed_model_path = addEmbedModelPath.trim();
      }

      const response = await api.addIndex(requestData);

      if (response.success) {
        toast.success(
          `Added ${response.data.added_chunks} chunks in ${response.data.time_cost?.toFixed(2)}s`
        );
        setAddCollectionName('');
        setAddDocsPaths([]);
        setAddEmbedModelPath('');
        loadCollections();
      } else {
        toast.error('Failed to add to index: ' + response.message);
      }
    } finally {
      setBuildLoading(false);
    }
  };

  const handleDeleteCollection = async (name: string) => {
    if (!confirm(`Are you sure you want to delete "${name}"? This cannot be undone.`)) {
      return;
    }

    setLoading(true);
    try {
      const response = await api.deleteCollection(name);
      if (response.success) {
        toast.success('Collection deleted');
        loadCollections();
      } else {
        toast.error('Failed to delete: ' + response.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const openManageDocuments = async (collectionName: string) => {
    setSelectedCollection(collectionName);
    setDeleteDialogOpen(true);
    setDeleteFilterType('filepath');
    setSelectedFilepath('');
    setSelectedDocId('');
    
    // Load documents using inspect API
    setLoadingDocs(true);
    try {
      const response = await api.inspectCollections();
      if (response.success) {
        // Find the specific collection
        const collection = response.data.collections?.find(
          (c: any) => c.collection_name === collectionName
        );
        
        if (collection) {
          // Extract unique filepaths and doc_ids from dynamic_fields
          // Since inspect doesn't give us all documents, we'll use the dynamic fields
          // to show what metadata fields are available
          const filepaths: string[] = [];
          const docIds: string[] = [];
          
          // Check if there are sample_data available in the response
          // If not, we'll just show the available fields
          setAvailableFilepaths(filepaths);
          setAvailableDocIds(docIds);
          
          toast.info('Note: Use inspect page to view available filepaths and doc_ids');
        } else {
          toast.error('Collection not found');
        }
      } else {
        toast.error('Failed to load collection info: ' + response.message);
      }
    } finally {
      setLoadingDocs(false);
    }
  };

  const handleDeleteDocuments = async () => {
    if (!selectedCollection) return;
    
    const requestData: any = {};
    
    if (deleteFilterType === 'filepath' && selectedFilepath) {
      requestData.filepath = selectedFilepath;
    } else if (deleteFilterType === 'doc_id' && selectedDocId) {
      requestData.doc_ids = [selectedDocId]; // API expects an array
    } else {
      toast.error('Please select a file or document ID');
      return;
    }

    const confirmMsg = deleteFilterType === 'filepath'
      ? `Delete all documents with filepath "${selectedFilepath}"?`
      : `Delete all documents with ID "${selectedDocId}"?`;
    
    if (!confirm(confirmMsg)) return;

    setLoadingDocs(true);
    try {
      const response = await api.deleteDocumentsByMetadata(selectedCollection, requestData);

      if (response.success) {
        toast.success('Documents deleted successfully');
        setDeleteDialogOpen(false);
        loadCollections();
      } else {
        toast.error('Failed to delete documents: ' + response.message);
      }
    } finally {
      setLoadingDocs(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Vector Index Management</h1>
          <p className="text-slate-600">Build and manage Milvus vector database</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Build Index */}
          <Card className="p-6">
            <h2 className="font-bold mb-4 flex items-center gap-2">
              <Plus className="w-5 h-5" />
              Build Index
            </h2>
            <div className="space-y-4">
              <div>
                <Label>Collection Name</Label>
                <Input
                  value={collectionName}
                  onChange={(e) => setCollectionName(e.target.value)}
                  placeholder="my_collection"
                />
              </div>
              <div>
                <Label>Documents Paths</Label>
                <div className="flex gap-2">
                  <Input
                    value={tempDocPath}
                    onChange={(e) => setTempDocPath(e.target.value)}
                    placeholder="/path/to/chunks.json or click + to browse"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && tempDocPath.trim()) {
                        setDocsPaths([...docsPaths, tempDocPath.trim()]);
                        setTempDocPath('');
                      }
                    }}
                  />
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".json"
                    className="hidden"
                    onChange={(e) => {
                      const files = Array.from(e.target.files || []);
                      const paths = files.map(f => f.name);
                      setDocsPaths([...docsPaths, ...paths]);
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => {
                      if (tempDocPath.trim()) {
                        setDocsPaths([...docsPaths, tempDocPath.trim()]);
                        setTempDocPath('');
                      } else {
                        fileInputRef.current?.click();
                      }
                    }}
                    size="sm"
                    variant="outline"
                  >
                    {tempDocPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                  </Button>
                </div>
                {docsPaths.length > 0 && (
                  <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                    <div className="p-2 space-y-1">
                      {docsPaths.map((path, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200"
                        >
                          <span className="text-xs font-mono truncate flex-1">{path}</span>
                          <Button
                            type="button"
                            onClick={() => setDocsPaths(docsPaths.filter((_, i) => i !== idx))}
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
                <Label>Embedding Model Path (Optional)</Label>
                <Input
                  value={embedModelPath}
                  onChange={(e) => setEmbedModelPath(e.target.value)}
                  placeholder="Optional: /path/to/embedding/model"
                />
              </div>
              <div>
                <Label>Embedding Dimension (Optional)</Label>
                <Input
                  type="number"
                  value={embedDim}
                  onChange={(e) => setEmbedDim(e.target.value)}
                  placeholder="e.g., 768"
                />
              </div>
              <div>
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                />
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div>
                  <Label className="text-sm">Enable Sparse (BM25)</Label>
                  <p className="text-xs text-slate-500">For Hybrid Search</p>
                </div>
                <Switch
                  checked={enableSparse}
                  onCheckedChange={setEnableSparse}
                />
              </div>
              <Button
                onClick={handleBuildIndex}
                disabled={buildLoading}
                className="w-full"
              >
                {buildLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Building...
                  </>
                ) : (
                  'Build Index'
                )}
              </Button>
            </div>
          </Card>

          {/* Add to Index */}
          <Card className="p-6">
            <h2 className="font-bold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5" />
              Add to Index
            </h2>
            <div className="space-y-4">
              <div>
                <Label>Collection Name</Label>
                <Input
                  value={addCollectionName}
                  onChange={(e) => setAddCollectionName(e.target.value)}
                  placeholder="existing_collection"
                />
              </div>
              <div>
                <Label>Documents Paths</Label>
                <div className="flex gap-2">
                  <Input
                    value={tempAddDocPath}
                    onChange={(e) => setTempAddDocPath(e.target.value)}
                    placeholder="/path/to/new_chunks.json or click + to browse"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && tempAddDocPath.trim()) {
                        setAddDocsPaths([...addDocsPaths, tempAddDocPath.trim()]);
                        setTempAddDocPath('');
                      }
                    }}
                  />
                  <input
                    ref={addFileInputRef}
                    type="file"
                    multiple
                    accept=".json"
                    className="hidden"
                    onChange={(e) => {
                      const files = Array.from(e.target.files || []);
                      const paths = files.map(f => f.name);
                      setAddDocsPaths([...addDocsPaths, ...paths]);
                      if (addFileInputRef.current) addFileInputRef.current.value = '';
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => {
                      if (tempAddDocPath.trim()) {
                        setAddDocsPaths([...addDocsPaths, tempAddDocPath.trim()]);
                        setTempAddDocPath('');
                      } else {
                        addFileInputRef.current?.click();
                      }
                    }}
                    size="sm"
                    variant="outline"
                  >
                    {tempAddDocPath.trim() ? <Plus className="w-4 h-4" /> : <FolderOpen className="w-4 h-4" />}
                  </Button>
                </div>
                {addDocsPaths.length > 0 && (
                  <ScrollArea className="h-20 mt-2 rounded-md border border-slate-200 bg-slate-50">
                    <div className="p-2 space-y-1">
                      {addDocsPaths.map((path, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between gap-2 p-1 px-2 bg-white rounded border border-slate-200"
                        >
                          <span className="text-xs font-mono truncate flex-1">{path}</span>
                          <Button
                            type="button"
                            onClick={() => setAddDocsPaths(addDocsPaths.filter((_, i) => i !== idx))}
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
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  value={addBatchSize}
                  onChange={(e) => setAddBatchSize(parseInt(e.target.value))}
                />
              </div>
              <div>
                <Label>Embedding Model Path (Optional)</Label>
                <Input
                  value={addEmbedModelPath}
                  onChange={(e) => setAddEmbedModelPath(e.target.value)}
                  placeholder="Optional: /path/to/embedding/model"
                />
                <div className="mt-2 space-y-1">
                  <p className="text-xs text-slate-500 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Vector type (sparse/dense) will follow original collection
                  </p>
                  <p className="text-xs text-slate-500 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Embedding dimension must match original collection
                  </p>
                </div>
              </div>
              <Button
                onClick={handleAddIndex}
                disabled={buildLoading}
                className="w-full"
                variant="outline"
              >
                {buildLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Adding...
                  </>
                ) : (
                  'Add to Index'
                )}
              </Button>
            </div>
          </Card>

          {/* Inspect */}
          <Card className="p-6">
            <h2 className="font-bold mb-4 flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Collection Details
            </h2>
            <p className="text-sm text-slate-600 mb-4">
              View collection schema and dynamic fields
            </p>
            <Dialog>
              <DialogTrigger asChild>
                <Button
                  onClick={loadInspectData}
                  disabled={loading}
                  className="w-full"
                  variant="outline"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Inspect Collections
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[80vh]">
                <DialogHeader>
                  <DialogTitle>Collection Details</DialogTitle>
                </DialogHeader>
                {inspectData && (
                  <ScrollArea className="h-[60vh]">
                    <div className="space-y-6">
                      {inspectData.collections?.map((coll: any, idx: number) => (
                        <Card key={idx} className="p-4">
                          <h3 className="font-bold mb-2">{coll.collection_name}</h3>
                          <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                            <div>
                              <span className="text-slate-500">URI:</span>{' '}
                              <span className="font-mono text-xs">{coll.uri}</span>
                            </div>
                            <div>
                              <span className="text-slate-500">Size:</span>{' '}
                              {formatBytes(coll.size_bytes)}
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h4 className="text-sm font-medium mb-2">Predefined Fields</h4>
                              <div className="flex flex-wrap gap-1">
                                {coll.predefined_fields?.map((field: string) => (
                                  <Badge key={field} variant="outline">
                                    {field}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            <div>
                              <h4 className="text-sm font-medium mb-2">Dynamic Fields</h4>
                              <div className="flex flex-wrap gap-1">
                                {coll.dynamic_fields?.map((field: string) => (
                                  <Badge key={field} className="bg-purple-100 text-purple-800">
                                    {field}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </DialogContent>
            </Dialog>
          </Card>
        </div>

        {/* Collections Table */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-bold flex items-center gap-2">
              <Database className="w-5 h-5" />
              Collections
            </h2>
            <Button
              onClick={loadCollections}
              disabled={loading}
              variant="outline"
              size="sm"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
            </div>
          ) : collections.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              <Database className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No collections yet</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Collection Name</TableHead>
                  <TableHead>Database File</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {collections.map((collection) => (
                  <TableRow key={collection.name}>
                    <TableCell className="font-medium">{collection.name}</TableCell>
                    <TableCell className="font-mono text-xs text-slate-500">
                      {collection.db_file}
                    </TableCell>
                    <TableCell>{formatBytes(collection.size_bytes)}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          onClick={() => openManageDocuments(collection.name)}
                          variant="ghost"
                          size="sm"
                          className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                        >
                          <FileText className="w-4 h-4" />
                        </Button>
                        <Button
                          onClick={() => handleDeleteCollection(collection.name)}
                          variant="ghost"
                          size="sm"
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Card>

        {/* Delete Documents Dialog */}
        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Manage Documents - {selectedCollection}</DialogTitle>
            </DialogHeader>

            <div className="space-y-4">
              <div>
                <Label>Delete by</Label>
                <Select
                  value={deleteFilterType}
                  onValueChange={(value: 'filepath' | 'doc_id') => {
                    setDeleteFilterType(value);
                    setSelectedFilepath('');
                    setSelectedDocId('');
                  }}
                >
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="filepath">File Path (Priority)</SelectItem>
                    <SelectItem value="doc_id">Document ID</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {deleteFilterType === 'filepath' ? (
                <div>
                  <Label>File Path to Delete</Label>
                  <Input
                    value={selectedFilepath}
                    onChange={(e) => setSelectedFilepath(e.target.value)}
                    placeholder="e.g., /path/to/document.pdf"
                    className="mt-1"
                  />
                  {selectedFilepath && (
                    <p className="text-xs text-slate-500 mt-2">
                      ⚠️ This will delete all documents with filepath: <strong>{selectedFilepath}</strong>
                    </p>
                  )}
                </div>
              ) : (
                <div>
                  <Label>Document ID to Delete</Label>
                  <Input
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    placeholder="e.g., doc_123"
                    className="mt-1"
                  />
                  {selectedDocId && (
                    <p className="text-xs text-slate-500 mt-2">
                      ⚠️ This will delete all documents with ID: <strong>{selectedDocId}</strong>
                    </p>
                  )}
                </div>
              )}

              {/* Info */}
              <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
                <p className="text-xs text-amber-800">
                  💡 Tip: Use the "Inspect Collections" dialog to view available filepaths and doc_ids in your collection.
                </p>
              </div>
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setDeleteDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteDocuments}
                disabled={loadingDocs || (!selectedFilepath && !selectedDocId)}
              >
                {loadingDocs ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete Documents
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
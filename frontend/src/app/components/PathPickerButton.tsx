import { useEffect, useMemo, useState } from 'react';
import { FolderOpen, ChevronUp, FileText, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from './ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Input } from './ui/input';
import { ScrollArea } from './ui/scroll-area';
import { api, type FileBrowserEntry } from '../utils/api';

type PathPickerMode = 'file' | 'directory';

interface PathPickerButtonProps {
  mode: PathPickerMode;
  value?: string;
  onSelect: (path: string) => void;
  title?: string;
  description?: string;
  allowedExtensions?: string[];
  buttonVariant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'destructive' | 'link';
  buttonSize?: 'default' | 'sm' | 'lg' | 'icon';
  className?: string;
}

const normalizeExtension = (value: string) => {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed) return '';
  return trimmed.startsWith('.') ? trimmed : `.${trimmed}`;
};

const joinPath = (directory: string, basename: string) => {
  const trimmedDirectory = directory.trim();
  const trimmedBasename = basename.trim();
  if (!trimmedDirectory) return trimmedBasename;
  if (!trimmedBasename) return trimmedDirectory;
  const separator = trimmedDirectory.includes('\\') ? '\\' : '/';
  return trimmedDirectory.endsWith('/') || trimmedDirectory.endsWith('\\')
    ? `${trimmedDirectory}${trimmedBasename}`
    : `${trimmedDirectory}${separator}${trimmedBasename}`;
};

const getBasename = (value: string) => {
  const parts = value.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? '';
};

export function PathPickerButton({
  mode,
  value,
  onSelect,
  title,
  description,
  allowedExtensions,
  buttonVariant = 'outline',
  buttonSize = 'sm',
  className,
}: PathPickerButtonProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [currentPath, setCurrentPath] = useState('');
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [roots, setRoots] = useState<string[]>([]);
  const [entries, setEntries] = useState<FileBrowserEntry[]>([]);
  const [customPath, setCustomPath] = useState('');

  const normalizedExtensions = useMemo(
    () => (allowedExtensions ?? []).map(normalizeExtension).filter(Boolean),
    [allowedExtensions],
  );

  const visibleEntries = useMemo(() => {
    if (mode === 'directory') {
      return entries.filter((entry) => entry.is_dir);
    }
    if (!normalizedExtensions.length) {
      return entries;
    }
    return entries.filter((entry) => {
      if (entry.is_dir) return true;
      const lowerPath = entry.path.toLowerCase();
      return normalizedExtensions.some((extension) => lowerPath.endsWith(extension));
    });
  }, [entries, mode, normalizedExtensions]);

  const loadDirectory = async (path?: string) => {
    setLoading(true);
    try {
      const response = await api.browseFiles(path);
      if (!response.success) {
        toast.error(`Failed to open path: ${response.message}`);
        return;
      }
      setCurrentPath(response.data.current_path);
      setParentPath(response.data.parent_path ?? null);
      setRoots(response.data.roots ?? []);
      setEntries(response.data.entries ?? []);
      setCustomPath(response.data.current_path);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    void loadDirectory(value);
  }, [open, value]);

  const handleSelectDirectory = () => {
    if (mode !== 'directory') return;
    const basename = getBasename(value ?? '');
    onSelect(basename && (value ?? '').includes('.') ? joinPath(currentPath, basename) : currentPath);
    setOpen(false);
  };

  const handleOpenCustomPath = async () => {
    const trimmed = customPath.trim();
    if (!trimmed) return;
    await loadDirectory(trimmed);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button type="button" variant={buttonVariant} size={buttonSize} className={className}>
          <FolderOpen className="w-4 h-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="w-[min(96vw,48rem)] max-w-[48rem] overflow-hidden">
        <DialogHeader>
          <DialogTitle>{title ?? (mode === 'directory' ? 'Select Directory' : 'Select File')}</DialogTitle>
        </DialogHeader>
        {description ? <p className="text-sm text-slate-500">{description}</p> : null}

        <div className="flex min-w-0 items-center gap-2">
          <div className="min-w-0 flex-1">
            <Input
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
              placeholder="Enter a path to open"
              className="w-full"
            />
          </div>
          <Button type="button" variant="outline" className="shrink-0" onClick={() => void handleOpenCustomPath()}>
            Open
          </Button>
          <Button
            type="button"
            variant="outline"
            className="shrink-0"
            onClick={() => void loadDirectory(currentPath)}
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          {roots.map((root) => (
            <Button key={root} type="button" variant="outline" size="sm" onClick={() => void loadDirectory(root)}>
              {root}
            </Button>
          ))}
        </div>

        <div className="flex min-w-0 items-center gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
          <span className="min-w-0 flex-1 truncate font-mono">{currentPath || 'Loading...'}</span>
          <div className="flex shrink-0 gap-2">
            {parentPath ? (
              <Button type="button" variant="outline" size="sm" className="shrink-0" onClick={() => void loadDirectory(parentPath)}>
                <ChevronUp className="w-4 h-4 mr-1" />
                Up
              </Button>
            ) : null}
            {mode === 'directory' ? (
              <Button type="button" size="sm" className="shrink-0" onClick={handleSelectDirectory} disabled={!currentPath}>
                Select This Folder
              </Button>
            ) : null}
          </div>
        </div>

        <ScrollArea className="h-[420px] rounded-md border border-slate-200">
          <div className="p-2 space-y-1">
            {visibleEntries.map((entry) => (
              <div
                key={entry.path}
                className="flex min-w-0 items-center gap-3 rounded-md border border-slate-200 bg-white px-3 py-2"
              >
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-center gap-2 text-left"
                  onClick={() => {
                    if (entry.is_dir) {
                      void loadDirectory(entry.path);
                      return;
                    }
                    if (mode === 'file') {
                      onSelect(entry.path);
                      setOpen(false);
                    }
                  }}
                >
                  {entry.is_dir ? (
                    <FolderOpen className="w-4 h-4 shrink-0 text-amber-600" />
                  ) : (
                    <FileText className="w-4 h-4 shrink-0 text-slate-500" />
                  )}
                  <span className="truncate font-mono text-sm">{entry.name}</span>
                </button>

                {entry.is_dir ? (
                  <Button type="button" variant="ghost" size="sm" className="shrink-0" onClick={() => void loadDirectory(entry.path)}>
                    Open
                  </Button>
                ) : (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="shrink-0"
                    disabled={mode !== 'file'}
                    onClick={() => {
                      onSelect(entry.path);
                      setOpen(false);
                    }}
                  >
                    Select
                  </Button>
                )}
              </div>
            ))}
            {!loading && visibleEntries.length === 0 ? (
              <div className="px-3 py-8 text-center text-sm text-slate-400">
                {mode === 'directory' ? 'No folders in this directory.' : 'No matching entries in this directory.'}
              </div>
            ) : null}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

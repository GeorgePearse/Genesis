import { useEffect, useState } from 'react';
import { useGenesis } from '../context/GenesisContext';
import { 
  Calculator, 
  Code, 
  FileText, 
  GitBranch, 
  GitGraph, 
  Layout, 
  RefreshCw, 
  Database,
  Activity,
  Map as MapIcon,
  BarChart2,
  Cpu,
  Zap,
  CheckCircle,
  FileDiff,
  Terminal,
  Brain,
  History
} from 'lucide-react';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command"

export default function CommandMenu() {
  const { 
    state, 
    loadDatabase, 
    setLeftTab, 
    setRightTab, 
    refreshData, 
    setAutoRefresh 
  } = useGenesis();
  
  const [open, setOpen] = useState(false);

  // Toggle the menu when ⌘K is pressed
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  const runCommand = (command: () => void) => {
    setOpen(false);
    command();
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Left Panel Views">
          <CommandItem onSelect={() => runCommand(() => setLeftTab('tree-view'))}>
            <GitGraph className="mr-2 h-4 w-4" />
            <span>Tree View</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('table-view'))}>
            <Layout className="mr-2 h-4 w-4" />
            <span>Programs Table</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('metrics-view'))}>
            <BarChart2 className="mr-2 h-4 w-4" />
            <span>Metrics</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('embeddings-view'))}>
            <Cpu className="mr-2 h-4 w-4" />
            <span>Embeddings</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('clusters-view'))}>
            <Activity className="mr-2 h-4 w-4" />
            <span>Clusters</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('islands-view'))}>
            <MapIcon className="mr-2 h-4 w-4" />
            <span>Islands</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('model-posteriors-view'))}>
            <Brain className="mr-2 h-4 w-4" />
            <span>LLM Posteriors</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setLeftTab('best-path-view'))}>
            <History className="mr-2 h-4 w-4" />
            <span>Path to Best</span>
          </CommandItem>
        </CommandGroup>

        <CommandGroup heading="Right Panel Views">
          <CommandItem onSelect={() => runCommand(() => setRightTab('meta-info'))}>
            <FileText className="mr-2 h-4 w-4" />
            <span>Meta Info</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('pareto-front'))}>
            <GitBranch className="mr-2 h-4 w-4" />
            <span>Pareto Front</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('scratchpad'))}>
            <Terminal className="mr-2 h-4 w-4" />
            <span>Scratchpad</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('node-details'))}>
            <CheckCircle className="mr-2 h-4 w-4" />
            <span>Node Details</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('code-viewer'))}>
            <Code className="mr-2 h-4 w-4" />
            <span>Code Viewer</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('diff-viewer'))}>
            <FileDiff className="mr-2 h-4 w-4" />
            <span>Diff Viewer</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('evaluation'))}>
            <Calculator className="mr-2 h-4 w-4" />
            <span>Evaluation Results</span>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setRightTab('llm-result'))}>
            <Brain className="mr-2 h-4 w-4" />
            <span>LLM Output</span>
          </CommandItem>
        </CommandGroup>

        <CommandGroup heading="Actions">
          <CommandItem onSelect={() => runCommand(() => refreshData())}>
            <RefreshCw className="mr-2 h-4 w-4" />
            <span>Refresh Data</span>
            <CommandShortcut>R</CommandShortcut>
          </CommandItem>
          <CommandItem onSelect={() => runCommand(() => setAutoRefresh(!state.autoRefreshEnabled))}>
            <Zap className="mr-2 h-4 w-4" />
            <span>{state.autoRefreshEnabled ? 'Disable' : 'Enable'} Auto-Refresh</span>
          </CommandItem>
        </CommandGroup>

        <CommandGroup heading="Databases">
          {state.databases.map((db) => {
             // Create a readable label
             const parts = db.path.split('/');
             const task = parts.length >= 3 ? parts[parts.length - 3] : 'Unknown';
             const name = parts.length >= 2 ? parts[parts.length - 2] : 'Unknown';
             return (
               <CommandItem 
                 key={db.path} 
                 onSelect={() => runCommand(() => loadDatabase(db.path))}
                 data-selected={state.currentDbPath === db.path ? "true" : "false"}
               >
                 <Database className="mr-2 h-4 w-4" />
                 <span>{task} / {name}</span>
               </CommandItem>
             );
          })}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}

import { Command } from 'cmdk';
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
  Search,
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
import './CommandMenu.css';

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
    <Command.Dialog open={open} onOpenChange={setOpen} label="Global Command Menu">
      <div className="search-icon">
        <Search />
      </div>
      <Command.Input placeholder="Type a command or search..." />
      <Command.List>
        <Command.Empty>No results found.</Command.Empty>

        <Command.Group heading="Left Panel Views">
          <Command.Item onSelect={() => runCommand(() => setLeftTab('tree-view'))}>
            <GitGraph />
            <span>Tree View</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('table-view'))}>
            <Layout />
            <span>Programs Table</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('metrics-view'))}>
            <BarChart2 />
            <span>Metrics</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('embeddings-view'))}>
            <Cpu />
            <span>Embeddings</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('clusters-view'))}>
            <Activity />
            <span>Clusters</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('islands-view'))}>
            <MapIcon />
            <span>Islands</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('model-posteriors-view'))}>
            <Brain />
            <span>LLM Posteriors</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setLeftTab('best-path-view'))}>
            <History />
            <span>Path to Best</span>
          </Command.Item>
        </Command.Group>

        <Command.Group heading="Right Panel Views">
          <Command.Item onSelect={() => runCommand(() => setRightTab('meta-info'))}>
            <FileText />
            <span>Meta Info</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('pareto-front'))}>
            <GitBranch />
            <span>Pareto Front</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('scratchpad'))}>
            <Terminal />
            <span>Scratchpad</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('node-details'))}>
            <CheckCircle />
            <span>Node Details</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('code-viewer'))}>
            <Code />
            <span>Code Viewer</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('diff-viewer'))}>
            <FileDiff />
            <span>Diff Viewer</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('evaluation'))}>
            <Calculator />
            <span>Evaluation Results</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setRightTab('llm-result'))}>
            <Brain />
            <span>LLM Output</span>
          </Command.Item>
        </Command.Group>

        <Command.Group heading="Actions">
          <Command.Item onSelect={() => runCommand(() => refreshData())}>
            <RefreshCw />
            <span>Refresh Data</span>
            <span className="cmdk-kbd">R</span>
          </Command.Item>
          <Command.Item onSelect={() => runCommand(() => setAutoRefresh(!state.autoRefreshEnabled))}>
            <Zap />
            <span>{state.autoRefreshEnabled ? 'Disable' : 'Enable'} Auto-Refresh</span>
          </Command.Item>
        </Command.Group>

        <Command.Group heading="Databases">
          {state.databases.map((db) => {
             // Create a readable label
             const parts = db.path.split('/');
             const task = parts.length >= 3 ? parts[parts.length - 3] : 'Unknown';
             const name = parts.length >= 2 ? parts[parts.length - 2] : 'Unknown';
             return (
               <Command.Item 
                 key={db.path} 
                 onSelect={() => runCommand(() => loadDatabase(db.path))}
                 data-selected={state.currentDbPath === db.path ? "true" : "false"}
               >
                 <Database />
                 <span>{task} / {name}</span>
               </Command.Item>
             );
          })}
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}

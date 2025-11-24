import { Database } from 'lucide-react';
import { useGenesis } from '../context/GenesisContext';

// Import existing panel components
import TreeView from './left-panel/TreeView';
import ProgramsTable from './left-panel/ProgramsTable';
import MetricsView from './left-panel/MetricsView';
import EmbeddingsView from './left-panel/EmbeddingsView';
import ClustersView from './left-panel/ClustersView';
import IslandsView from './left-panel/IslandsView';
import ModelPosteriorsView from './left-panel/ModelPosteriorsView';
import BestPathView from './left-panel/BestPathView';
import MetaInfoPanel from './right-panel/MetaInfoPanel';
import ParetoFrontPanel from './right-panel/ParetoFrontPanel';
import ScratchpadPanel from './right-panel/ScratchpadPanel';
import NodeDetailsPanel from './right-panel/NodeDetailsPanel';
import CodeViewerPanel from './right-panel/CodeViewerPanel';
import DiffViewerPanel from './right-panel/DiffViewerPanel';
import EvaluationPanel from './right-panel/EvaluationPanel';
import LLMResultPanel from './right-panel/LLMResultPanel';

interface MainContentProps {
  activeTab: string;
}

export default function MainContent({ activeTab }: MainContentProps) {
  const { state } = useGenesis();

  const renderContent = () => {
    // If no data loaded, show empty state
    if (state.programs.length === 0) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="text-center max-w-md">
            <div className="mb-6 p-8 bg-gray-900 rounded-lg border border-gray-800">
              <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-300 mb-2">
                No Database Selected
              </h3>
              <p className="text-sm text-gray-500 mb-6">
                Select a database to view evolution results and begin analysis.
              </p>
              <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium text-sm transition-colors">
                Load Database
              </button>
            </div>
          </div>
        </div>
      );
    }

    // Render the appropriate view based on active tab
    switch (activeTab) {
      case 'Tree':
        return <TreeView />;
      case 'Programs':
        return <ProgramsTable />;
      case 'Metrics':
        return <MetricsView />;
      case 'Embeddings':
        return <EmbeddingsView />;
      case 'Clusters':
        return <ClustersView />;
      case 'Islands':
        return <IslandsView />;
      case 'LLM Posterior':
        return <ModelPosteriorsView />;
      case 'Path → Best':
        return <BestPathView />;
      case 'Meta':
        return <MetaInfoPanel />;
      case 'Pareto Front':
        return <ParetoFrontPanel />;
      case 'Scratchpad':
        return <ScratchpadPanel />;
      case 'Node':
        return <NodeDetailsPanel />;
      case 'Code':
        return <CodeViewerPanel />;
      case 'Diff':
        return <DiffViewerPanel />;
      case 'Evaluation':
        return <EvaluationPanel />;
      case 'LLM Result':
        return <LLMResultPanel />;
      default:
        return <TreeView />;
    }
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Content Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-medium text-white">{activeTab}</h2>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
              Export
            </button>
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white hover:bg-gray-800 rounded transition-colors">
              Settings
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8 overflow-auto">{renderContent()}</div>
    </div>
  );
}

import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { useGenesis } from '../../context/GenesisContext';
import './MetricsView.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function MetricsView() {
  const { state } = useGenesis();
  const { programs } = state;

  const metricsData = useMemo(() => {
    if (programs.length === 0) return null;

    // Group by generation
    const byGeneration = new Map<number, typeof programs>();
    programs.forEach((p) => {
      if (!byGeneration.has(p.generation)) {
        byGeneration.set(p.generation, []);
      }
      byGeneration.get(p.generation)!.push(p);
    });

    const generations = [...byGeneration.keys()].sort((a, b) => a - b);
    const maxScores: number[] = [];
    const avgScores: number[] = [];
    const cumulativeCosts: number[] = [];
    let runningCost = 0;

    generations.forEach((gen) => {
      const genPrograms = byGeneration.get(gen)!;
      const correctPrograms = genPrograms.filter(
        (p) => p.correct && p.combined_score !== null
      );

      if (correctPrograms.length > 0) {
        const scores = correctPrograms.map((p) => p.combined_score as number);
        maxScores.push(Math.max(...scores));
        avgScores.push(scores.reduce((a, b) => a + b, 0) / scores.length);
      } else {
        maxScores.push(maxScores[maxScores.length - 1] || 0);
        avgScores.push(avgScores[avgScores.length - 1] || 0);
      }

      genPrograms.forEach((p) => {
        runningCost +=
          (p.metadata.api_cost || 0) +
          (p.metadata.embed_cost || 0) +
          (p.metadata.novelty_cost || 0) +
          (p.metadata.meta_cost || 0);
      });
      cumulativeCosts.push(runningCost);
    });

    return { generations, maxScores, avgScores, cumulativeCosts };
  }, [programs]);

  if (!metricsData) {
    return (
      <div className="metrics-view empty">
        <p>No data available</p>
      </div>
    );
  }

  const scoreChartData = {
    labels: metricsData.generations.map((g) => `Gen ${g}`),
    datasets: [
      {
        label: 'Max Score',
        data: metricsData.maxScores,
        borderColor: '#2ecc71',
        backgroundColor: 'rgba(46, 204, 113, 0.1)',
        tension: 0.1,
      },
      {
        label: 'Avg Score',
        data: metricsData.avgScores,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.1)',
        tension: 0.1,
      },
    ],
  };

  const costChartData = {
    labels: metricsData.generations.map((g) => `Gen ${g}`),
    datasets: [
      {
        label: 'Cumulative Cost ($)',
        data: metricsData.cumulativeCosts,
        borderColor: '#e74c3c',
        backgroundColor: 'rgba(231, 76, 60, 0.1)',
        fill: true,
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  return (
    <div className="metrics-view">
      <h4>Metrics Over Generations</h4>

      <div className="chart-container">
        <h5>Performance Score</h5>
        <div className="chart-wrapper">
          <Line data={scoreChartData} options={chartOptions} />
        </div>
      </div>

      <div className="chart-container">
        <h5>Cumulative Cost</h5>
        <div className="chart-wrapper">
          <Line data={costChartData} options={chartOptions} />
        </div>
      </div>
    </div>
  );
}

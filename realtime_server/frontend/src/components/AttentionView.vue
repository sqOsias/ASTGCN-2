<template>
  <div class="h-full flex">
    <!-- Left: Attention Heatmap -->
    <div class="flex-1 flex flex-col p-4">
      <div class="cyber-panel flex-1 flex flex-col">
        <div class="cyber-panel-header">
          <el-icon><Grid /></el-icon>
          自适应空间注意力矩阵 (Adaptive Adjacency Matrix)
          <span class="ml-auto text-xs text-slate-500">点击热力方块查看节点关联</span>
        </div>
        
        <div class="flex-1 relative">
          <v-chart 
            ref="heatmapRef"
            class="w-full h-full" 
            :option="heatmapOption" 
            :autoresize="true"
            @click="handleHeatmapClick"
          />
          
          <!-- Loading Overlay -->
          <div 
            v-if="loading" 
            class="absolute inset-0 flex items-center justify-center bg-slate-900/80"
          >
            <div class="text-center">
              <el-icon :size="40" class="animate-spin text-cyan-400"><Loading /></el-icon>
              <div class="text-slate-400 mt-2">加载注意力矩阵...</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Bottom Info -->
      <div class="mt-3 flex gap-3">
        <div class="cyber-panel flex-1 p-3">
          <div class="text-xs text-slate-400 mb-2">矩阵说明</div>
          <p class="text-xs text-slate-500 leading-relaxed">
            该热力图展示了模型学习到的节点间空间依赖关系。颜色越红，表示两个节点在交通流传播中的关联越强。
            即使物理距离较远的节点，也可能因为交通流的传导效应而产生强关联。
          </p>
        </div>
        
        <div class="cyber-panel w-64 p-3">
          <div class="text-xs text-slate-400 mb-2">注意力统计</div>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span class="text-slate-500">矩阵维度:</span>
              <span class="text-cyan-400 ml-1">307 × 307</span>
            </div>
            <div>
              <span class="text-slate-500">非零值:</span>
              <span class="text-cyan-400 ml-1">{{ nonZeroCount.toLocaleString() }}</span>
            </div>
            <div>
              <span class="text-slate-500">最大值:</span>
              <span class="text-red-400 ml-1">{{ maxAttention.toFixed(4) }}</span>
            </div>
            <div>
              <span class="text-slate-500">均值:</span>
              <span class="text-yellow-400 ml-1">{{ avgAttention.toFixed(4) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Right: Node Relationship Panel -->
    <div class="w-96 border-l border-cyan-500/20 flex flex-col">
      <!-- Selected Pair Info -->
      <div class="cyber-panel m-3">
        <div class="cyber-panel-header">
          <el-icon><Connection /></el-icon>
          节点关联分析
        </div>
        <div class="p-4">
          <div v-if="selectedPair" class="space-y-4">
            <div class="flex items-center justify-center gap-4">
              <div class="text-center">
                <div class="w-16 h-16 rounded-full bg-cyan-500/20 flex items-center justify-center border-2 border-cyan-500">
                  <span class="text-cyan-400 font-bold">{{ selectedPair.source }}</span>
                </div>
                <div class="text-xs text-slate-400 mt-1">源节点</div>
              </div>
              
              <div class="flex flex-col items-center">
                <el-icon :size="24" class="text-orange-400"><Right /></el-icon>
                <div class="text-orange-400 font-mono text-sm">
                  {{ selectedPair.weight.toFixed(4) }}
                </div>
              </div>
              
              <div class="text-center">
                <div class="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center border-2 border-purple-500">
                  <span class="text-purple-400 font-bold">{{ selectedPair.target }}</span>
                </div>
                <div class="text-xs text-slate-400 mt-1">目标节点</div>
              </div>
            </div>
            
            <div class="bg-slate-800/50 rounded p-3 text-xs">
              <div class="text-cyan-400 font-bold mb-1">模型解释:</div>
              <p class="text-slate-400 leading-relaxed">
                当 Node_{{ selectedPair.source }} 的交通状态发生变化时，
                模型会以 <span class="text-orange-400">{{ (selectedPair.weight * 100).toFixed(2) }}%</span> 的权重
                将该变化传递给 Node_{{ selectedPair.target }}。
                这种隐式关联可能源于交通流的物理传播或时空模式的相似性。
              </p>
            </div>
          </div>
          
          <div v-else class="text-center text-slate-500 py-8">
            <el-icon :size="40"><Select /></el-icon>
            <div class="mt-2">点击热力图选择节点对</div>
          </div>
        </div>
      </div>
      
      <!-- Mini Graph -->
      <div class="cyber-panel m-3 flex-1 flex flex-col min-h-0">
        <div class="cyber-panel-header">
          <el-icon><Share /></el-icon>
          关联节点高亮
        </div>
        <div class="flex-1">
          <v-chart 
            ref="miniGraphRef"
            class="w-full h-full" 
            :option="miniGraphOption" 
            :autoresize="true"
          />
        </div>
      </div>
      
      <!-- Top Connections -->
      <div class="cyber-panel m-3">
        <div class="cyber-panel-header">
          <el-icon><Sort /></el-icon>
          最强关联 TOP 10
        </div>
        <div class="max-h-48 overflow-auto">
          <div 
            v-for="(conn, index) in topConnections" 
            :key="index"
            class="flex items-center gap-2 p-2 hover:bg-slate-800/50 cursor-pointer text-xs border-b border-slate-800"
            @click="selectPair(conn.source, conn.target, conn.weight)"
          >
            <span class="text-slate-500 w-4">{{ index + 1 }}</span>
            <span class="text-cyan-400">{{ conn.source }}</span>
            <el-icon class="text-slate-600"><Right /></el-icon>
            <span class="text-purple-400">{{ conn.target }}</span>
            <span class="ml-auto text-orange-400 font-mono">{{ conn.weight.toFixed(4) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, shallowRef, reactive } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart, GraphChart } from 'echarts/charts'
import { 
  TooltipComponent, 
  GridComponent, 
  VisualMapComponent,
  DataZoomComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer, 
  HeatmapChart, 
  GraphChart,
  TooltipComponent, 
  GridComponent, 
  VisualMapComponent,
  DataZoomComponent
])

const props = defineProps({
  topology: { type: Object, default: () => ({ nodes: [], edges: [] }) }
})

const emit = defineEmits(['node-pair-click'])

const heatmapRef = shallowRef(null)
const miniGraphRef = shallowRef(null)
const loading = ref(true)

const attentionMatrix = ref([])
const selectedPair = ref(null)
const topConnections = ref([])

// Statistics
const nonZeroCount = ref(0)
const maxAttention = ref(0)
const avgAttention = ref(0)

// Fetch attention matrix
const fetchAttentionMatrix = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/attention')
    const data = await res.json()
    
    // Process matrix for heatmap (sample for performance)
    const matrix = data.matrix
    const size = data.size
    
    // Calculate statistics
    let sum = 0
    let count = 0
    let max = 0
    const connections = []
    
    // Sample the matrix (every 3rd point for performance)
    const sampleRate = 3
    const heatmapData = []
    
    for (let i = 0; i < size; i += sampleRate) {
      for (let j = 0; j < size; j += sampleRate) {
        const val = matrix[i][j]
        if (val > 0.001) {
          heatmapData.push([Math.floor(i / sampleRate), Math.floor(j / sampleRate), val])
          sum += val
          count++
          if (val > max) max = val
          
          if (i !== j && val > 0.01) {
            connections.push({ source: i, target: j, weight: val })
          }
        }
      }
    }
    
    attentionMatrix.value = heatmapData
    nonZeroCount.value = count
    maxAttention.value = max
    avgAttention.value = count > 0 ? sum / count : 0
    
    // Sort and get top connections
    connections.sort((a, b) => b.weight - a.weight)
    topConnections.value = connections.slice(0, 10)
    
  } catch (e) {
    console.error('Error fetching attention matrix:', e)
    // Generate synthetic data for demo
    generateSyntheticAttention()
  } finally {
    loading.value = false
  }
}

const generateSyntheticAttention = () => {
  const size = 307
  const sampleRate = 3
  const sampledSize = Math.ceil(size / sampleRate)
  const heatmapData = []
  const connections = []
  let sum = 0
  let count = 0
  let max = 0
  
  for (let i = 0; i < sampledSize; i++) {
    for (let j = 0; j < sampledSize; j++) {
      // Create realistic attention pattern
      const distance = Math.abs(i - j)
      let val = Math.exp(-distance / 10) * 0.5 + Math.random() * 0.3
      
      // Add some strong connections
      if (Math.random() < 0.02) {
        val = 0.8 + Math.random() * 0.2
      }
      
      if (val > 0.1) {
        heatmapData.push([i, j, val])
        sum += val
        count++
        if (val > max) max = val
        
        if (i !== j) {
          connections.push({ 
            source: i * sampleRate, 
            target: j * sampleRate, 
            weight: val 
          })
        }
      }
    }
  }
  
  attentionMatrix.value = heatmapData
  nonZeroCount.value = count
  maxAttention.value = max
  avgAttention.value = count > 0 ? sum / count : 0
  
  connections.sort((a, b) => b.weight - a.weight)
  topConnections.value = connections.slice(0, 10)
}

// Heatmap option
const heatmapOption = computed(() => {
  const sampledSize = Math.ceil(307 / 3)
  
  return {
    backgroundColor: 'transparent',
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#e2e8f0' },
      formatter: (params) => {
        const [x, y, val] = params.data
        return `Node_${x * 3} → Node_${y * 3}<br/>Attention: ${val.toFixed(4)}`
      }
    },
    grid: {
      left: 60,
      right: 80,
      top: 20,
      bottom: 60
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: sampledSize }, (_, i) => i * 3),
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { 
        color: '#64748b', 
        fontSize: 8,
        interval: 9,
        rotate: 45
      }
    },
    yAxis: {
      type: 'category',
      data: Array.from({ length: sampledSize }, (_, i) => i * 3),
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { 
        color: '#64748b', 
        fontSize: 8,
        interval: 9
      }
    },
    visualMap: {
      min: 0,
      max: maxAttention.value || 1,
      calculable: true,
      orient: 'vertical',
      right: 10,
      top: 'center',
      textStyle: { color: '#64748b' },
      inRange: {
        color: ['#0f172a', '#1e3a5f', '#3b82f6', '#f97316', '#ef4444']
      }
    },
    series: [{
      name: 'Attention',
      type: 'heatmap',
      data: attentionMatrix.value,
      emphasis: {
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 1
        }
      }
    }]
  }
})

// Mini graph option
const miniGraphOption = computed(() => {
  if (!selectedPair.value) {
    // Show full graph dimmed
    return {
      backgroundColor: 'transparent',
      series: [{
        type: 'graph',
        layout: 'circular',
        data: [],
        links: [],
        roam: true
      }]
    }
  }
  
  const sourceId = selectedPair.value.source
  const targetId = selectedPair.value.target
  
  // Find related nodes from topology
  const relatedEdges = props.topology.edges.filter(e => 
    e.source === sourceId || e.target === sourceId ||
    e.source === targetId || e.target === targetId
  ).slice(0, 20)
  
  const relatedNodeIds = new Set([sourceId, targetId])
  relatedEdges.forEach(e => {
    relatedNodeIds.add(e.source)
    relatedNodeIds.add(e.target)
  })
  
  const nodes = Array.from(relatedNodeIds).map(id => ({
    id: String(id),
    name: `Node_${id}`,
    symbolSize: id === sourceId || id === targetId ? 30 : 15,
    itemStyle: {
      color: id === sourceId ? '#22d3ee' : id === targetId ? '#a855f7' : '#475569'
    },
    label: {
      show: id === sourceId || id === targetId,
      color: '#e2e8f0',
      fontSize: 10
    }
  }))
  
  const links = relatedEdges.map(e => ({
    source: String(e.source),
    target: String(e.target),
    lineStyle: {
      color: (e.source === sourceId && e.target === targetId) || 
             (e.source === targetId && e.target === sourceId) 
        ? '#f97316' : '#334155',
      width: (e.source === sourceId && e.target === targetId) || 
             (e.source === targetId && e.target === sourceId) 
        ? 3 : 1
    }
  }))
  
  // Add main connection
  links.push({
    source: String(sourceId),
    target: String(targetId),
    lineStyle: {
      color: '#f97316',
      width: 4,
      type: 'solid'
    }
  })
  
  return {
    backgroundColor: 'transparent',
    series: [{
      type: 'graph',
      layout: 'force',
      data: nodes,
      links: links,
      roam: true,
      force: {
        repulsion: 100,
        edgeLength: 50
      },
      emphasis: {
        focus: 'adjacency'
      }
    }]
  }
})

// Handle heatmap click
const handleHeatmapClick = (params) => {
  if (params.data) {
    const [x, y, weight] = params.data
    selectPair(x * 3, y * 3, weight)
  }
}

const selectPair = (source, target, weight) => {
  selectedPair.value = { source, target, weight }
  emit('node-pair-click', source, target)
}

onMounted(() => {
  fetchAttentionMatrix()
})
</script>

<template>
  <Splitpanes class="h-full bg-[#0a0e1a]">
    <Pane :size="75" :min-size="30">
    <!-- Left: Main Content -->
    <div class="h-full flex flex-col min-w-0">
      <!-- Top stats bar -->
      <div class="flex items-center gap-4 px-5 py-3 border-b border-slate-800/60">
        <div class="text-cyan-400 font-bold text-base tracking-wider">
          <span class="text-cyan-300">◈</span> 模型可解释性
        </div>
        <div class="flex gap-4 ml-4 text-xs">
          <div class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-cyan-500"></span>
            <span class="text-slate-500">矩阵维度</span>
            <span class="text-cyan-400 font-mono">{{ matrixSize }}×{{ matrixSize }}</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
            <span class="text-slate-500">非零关联</span>
            <span class="text-amber-400 font-mono">{{ nonZeroCount.toLocaleString() }}</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-red-500"></span>
            <span class="text-slate-500">最大权重</span>
            <span class="text-red-400 font-mono">{{ maxAttention.toFixed(4) }}</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            <span class="text-slate-500">平均权重</span>
            <span class="text-emerald-400 font-mono">{{ avgAttention.toFixed(6) }}</span>
          </div>
        </div>
        <!-- Node picker -->
        <div class="ml-auto flex items-center gap-2">
          <span class="text-xs text-slate-500">查看节点:</span>
          <el-input-number
            v-model="focusNode"
            :min="0" :max="Math.max(0, matrixSize - 1)"
            size="small"
            style="width: 100px"
            @change="onFocusNodeChange"
          />
        </div>
      </div>

      <!-- Main area: heatmap + network graph -->
      <Splitpanes class="flex-1" :horizontal="false">
        <Pane :size="60" :min-size="35">
        <!-- Heatmap -->
        <div class="h-full relative">
          <v-chart
            ref="heatmapRef"
            class="w-full h-full"
            :option="heatmapOption"
            :autoresize="true"
            @click="handleHeatmapClick"
          />
          <div
            v-if="loading"
            class="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10"
          >
            <div class="text-center">
              <el-icon :size="36" class="animate-spin text-cyan-400"><Loading /></el-icon>
              <div class="text-slate-400 mt-2 text-sm">加载注意力矩阵...</div>
            </div>
          </div>
        </div>

        </Pane>
        <Pane :size="40" :min-size="20">
        <!-- Network graph showing attention connections -->
        <div class="h-full relative">
          <v-chart
            ref="graphRef"
            class="w-full h-full"
            :option="graphOption"
            :autoresize="true"
          />
          <div class="absolute top-3 left-3 text-[10px] text-slate-600">
            拓扑图 · 注意力高亮
          </div>
        </div>
        </Pane>
      </Splitpanes>
    </div>
    </Pane>

    <Pane :size="25" :min-size="8">
    <!-- Right sidebar -->
    <Splitpanes horizontal class="h-full bg-[#080c16]">
      <Pane :size="30" :min-size="10">
        <!-- Node pair info -->
        <div class="h-full p-4 overflow-auto">
          <div class="text-xs text-slate-500 mb-3 flex items-center gap-2">
            <span class="w-1 h-3 bg-cyan-500 rounded"></span> 节点关联分析
          </div>
          <div v-if="selectedPair" class="space-y-3">
            <div class="flex items-center justify-center gap-3">
              <div class="text-center">
                <div class="w-14 h-14 rounded-xl bg-cyan-500/10 flex items-center justify-center border border-cyan-500/40">
                  <span class="text-cyan-400 font-bold text-lg">{{ selectedPair.source }}</span>
                </div>
                <div class="text-[10px] text-slate-500 mt-1">源节点</div>
              </div>
              <div class="flex flex-col items-center gap-1">
                <div class="w-12 h-px bg-gradient-to-r from-cyan-500 to-orange-500"></div>
                <div class="text-orange-400 font-mono text-sm font-bold">{{ selectedPair.weight.toFixed(4) }}</div>
                <div class="w-12 h-px bg-gradient-to-r from-orange-500 to-purple-500"></div>
              </div>
              <div class="text-center">
                <div class="w-14 h-14 rounded-xl bg-purple-500/10 flex items-center justify-center border border-purple-500/40">
                  <span class="text-purple-400 font-bold text-lg">{{ selectedPair.target }}</span>
                </div>
                <div class="text-[10px] text-slate-500 mt-1">目标节点</div>
              </div>
            </div>
            <div class="bg-slate-800/30 rounded-lg p-2.5 text-[11px] text-slate-400 leading-relaxed border border-slate-700/30">
              模型以 <span class="text-orange-400 font-bold">{{ (selectedPair.weight * 100).toFixed(2) }}%</span>
              的注意力权重关联这两个传感器。该权重反映了交通流传播中的空间依赖强度。
            </div>
          </div>
          <div v-else class="text-center py-6">
            <div class="text-2xl mb-2">🔍</div>
            <div class="text-xs text-slate-600">点击热力图或输入节点ID</div>
          </div>
        </div>
      </Pane>
      <Pane :size="45" :min-size="15">
        <!-- Node attention bar chart (top N neighbors) -->
        <div class="h-full p-4 overflow-hidden flex flex-col">
          <div class="text-xs text-slate-500 mb-2 flex items-center gap-2 flex-shrink-0">
            <span class="w-1 h-3 bg-orange-500 rounded"></span>
            节点 {{ focusNode }} 的注意力分布 (Top 15)
          </div>
          <div class="flex-1 min-h-0">
            <v-chart
              class="w-full h-full"
              :option="barChartOption"
              :autoresize="true"
              @click="handleBarClick"
            />
          </div>
        </div>
      </Pane>
      <Pane :size="25" :min-size="10">
        <!-- Top connections list -->
        <div class="h-full p-4 overflow-auto scrollbar-thin">
          <div class="text-xs text-slate-500 mb-2 flex items-center gap-2">
            <span class="w-1 h-3 bg-red-500 rounded"></span> 全局最强关联 TOP 10
          </div>
          <div
            v-for="(conn, i) in topConnections"
            :key="i"
            class="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-slate-800/40 cursor-pointer text-xs transition-colors"
            @click="selectPair(conn.source, conn.target, conn.weight)"
          >
            <span class="w-4 text-slate-600 font-mono">{{ i + 1 }}</span>
            <span class="text-cyan-400 font-mono w-8 text-right">{{ conn.source }}</span>
            <span class="text-slate-600">→</span>
            <span class="text-purple-400 font-mono w-8">{{ conn.target }}</span>
            <div class="flex-1 mx-2 h-1 rounded-full bg-slate-800 overflow-hidden">
              <div class="h-full rounded-full bg-gradient-to-r from-orange-500 to-red-500"
                   :style="{ width: (conn.weight / (topConnections[0]?.weight || 1) * 100) + '%' }"></div>
            </div>
            <span class="text-orange-400 font-mono w-12 text-right">{{ conn.weight.toFixed(4) }}</span>
          </div>
        </div>
      </Pane>
    </Splitpanes>
    </Pane>
  </Splitpanes>
</template>

<script setup>
import { ref, computed, onMounted, shallowRef, watch } from 'vue'
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart, GraphChart, BarChart } from 'echarts/charts'
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
  BarChart,
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
const graphRef = shallowRef(null)
const loading = ref(true)

const rawMatrix = ref(null)   // full 307x307 matrix
const attentionMatrix = ref([])
const selectedPair = ref(null)
const topConnections = ref([])
const focusNode = ref(0)
const focusNeighbors = ref([]) // top-N attention neighbors for focusNode
const matrixSize = ref(0)

// Statistics
const nonZeroCount = ref(0)
const maxAttention = ref(0)
const avgAttention = ref(0)

// ============== Fetch ==============
const fetchAttentionMatrix = async () => {
  loading.value = true
  try {
    const res = await fetch('/api/attention')
    const data = await res.json()

    const matrix = data.matrix
    const size = data.size
    matrixSize.value = size
    rawMatrix.value = matrix

    let sum = 0, count = 0, max = 0
    const connections = []
    const sampleRate = 3
    const heatmapData = []

    for (let i = 0; i < size; i += sampleRate) {
      for (let j = 0; j < size; j += sampleRate) {
        const val = matrix[i][j]
        if (val > 0.001) {
          heatmapData.push([Math.floor(i / sampleRate), Math.floor(j / sampleRate), val])
          sum += val; count++
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

    connections.sort((a, b) => b.weight - a.weight)
    topConnections.value = connections.slice(0, 10)

    // Initialize focus node neighbors
    updateFocusNeighbors()
  } catch (e) {
    console.error('Error fetching attention matrix:', e)
  } finally {
    loading.value = false
  }
}

// ============== Focus node attention distribution ==============
const updateFocusNeighbors = () => {
  if (!rawMatrix.value) return
  const row = rawMatrix.value[focusNode.value]
  if (!row) return
  const neighbors = row.map((w, j) => ({ node: j, weight: w }))
    .filter(n => n.node !== focusNode.value && n.weight > 0.001)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 15)
  focusNeighbors.value = neighbors
}

const onFocusNodeChange = () => {
  updateFocusNeighbors()
}

// ============== Node positions ==============
const nodePositions = computed(() => {
  const positions = {}
  if (props.topology.nodes && props.topology.nodes.length) {
    props.topology.nodes.forEach(n => {
      positions[n.id] = { x: n.x, y: n.y }
    })
  }
  return positions
})

// ============== Heatmap ==============
const heatmapOption = computed(() => {
  const sampleRate = 3
  const sampledSize = Math.ceil((matrixSize.value || 0) / sampleRate)
  return {
    backgroundColor: 'transparent',
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(10,14,26,0.95)',
      borderColor: 'rgba(34,211,238,0.3)',
      borderWidth: 1,
      textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: (params) => {
        const [x, y, val] = params.data
        return `<b>Node_${x * sampleRate} → Node_${y * sampleRate}</b><br/>注意力权重: <span style="color:#f97316">${val.toFixed(4)}</span>`
      }
    },
    grid: { left: 50, right: 70, top: 15, bottom: 50 },
    xAxis: {
      type: 'category',
      data: Array.from({ length: sampledSize }, (_, i) => i * sampleRate),
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#1e293b' } },
      axisLabel: { color: '#475569', fontSize: 8, interval: 9, rotate: 45 },
      name: '目标节点', nameLocation: 'center', nameGap: 35,
      nameTextStyle: { color: '#475569', fontSize: 10 }
    },
    yAxis: {
      type: 'category',
      data: Array.from({ length: sampledSize }, (_, i) => i * sampleRate),
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#1e293b' } },
      axisLabel: { color: '#475569', fontSize: 8, interval: 9 },
      name: '源节点', nameLocation: 'center', nameGap: 40,
      nameTextStyle: { color: '#475569', fontSize: 10 }
    },
    visualMap: {
      min: 0, max: maxAttention.value || 1,
      calculable: true, orient: 'vertical',
      right: 5, top: 'center',
      itemHeight: 180,
      textStyle: { color: '#475569', fontSize: 9 },
      inRange: { color: ['#0a0e1a', '#0c2340', '#1d4ed8', '#f59e0b', '#ef4444'] }
    },
    series: [{
      name: 'Attention',
      type: 'heatmap',
      data: attentionMatrix.value,
      progressive: 5000,
      emphasis: { itemStyle: { borderColor: '#22d3ee', borderWidth: 2 } }
    }]
  }
})

// ============== Network graph with attention highlight ==============
const graphOption = computed(() => {
  const focusSet = new Set(focusNeighbors.value.map(n => n.node))
  focusSet.add(focusNode.value)
  const neighborWeights = {}
  focusNeighbors.value.forEach(n => { neighborWeights[n.node] = n.weight })

  const nodes = []
  for (let i = 0; i < matrixSize.value; i++) {
    const pos = nodePositions.value[i] || { x: 400, y: 300 }
    const isFocus = i === focusNode.value
    const isNeighbor = neighborWeights[i] !== undefined
    const w = neighborWeights[i] || 0
    const dimmed = focusNeighbors.value.length > 0 && !focusSet.has(i)

    nodes.push({
      id: String(i),
      name: `Node_${i}`,
      x: pos.x, y: pos.y, fixed: true,
      symbolSize: isFocus ? 14 : isNeighbor ? Math.max(5, w * 60) : 3,
      itemStyle: {
        color: isFocus ? '#22d3ee'
          : isNeighbor ? `rgba(249,115,22,${Math.max(0.4, w * 3)})`
          : dimmed ? 'rgba(71,85,105,0.1)' : 'rgba(71,85,105,0.25)',
        borderColor: isFocus ? '#fff' : 'transparent',
        borderWidth: isFocus ? 2 : 0,
        shadowColor: isFocus ? 'rgba(34,211,238,0.8)' : 'transparent',
        shadowBlur: isFocus ? 15 : 0
      },
      label: {
        show: isFocus,
        formatter: `{a|${i}}`,
        rich: { a: { color: '#22d3ee', fontSize: 10, fontWeight: 'bold' } }
      },
      value: isFocus ? 1 : w
    })
  }

  const edges = []
  props.topology.edges.forEach(e => {
    const hasFocus = e.source === focusNode.value || e.target === focusNode.value
    const bothRelevant = focusSet.has(e.source) && focusSet.has(e.target)
    const dimmed = focusNeighbors.value.length > 0 && !bothRelevant

    edges.push({
      source: String(e.source), target: String(e.target),
      lineStyle: {
        color: hasFocus ? 'rgba(34,211,238,0.6)'
          : bothRelevant ? 'rgba(249,115,22,0.3)'
          : dimmed ? 'rgba(71,85,105,0.03)' : 'rgba(71,85,105,0.08)',
        width: hasFocus ? 2 : bothRelevant ? 1 : 0.3
      }
    })
  })

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(10,14,26,0.95)',
      borderColor: 'rgba(34,211,238,0.3)',
      borderWidth: 1,
      textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: (p) => {
        if (p.dataType === 'node') {
          const w = neighborWeights[parseInt(p.data.id)]
          return w !== undefined
            ? `<b>${p.data.name}</b><br/>注意力: <span style="color:#f97316">${w.toFixed(4)}</span>`
            : `<b>${p.data.name}</b>`
        }
        return ''
      }
    },
    series: [{
      type: 'graph', layout: 'none', animation: false,
      data: nodes, links: edges, roam: true, zoom: 1.1,
      emphasis: { focus: 'adjacency', lineStyle: { width: 3 } },
      label: { show: false },
      lineStyle: { curveness: 0.15 }
    }]
  }
})

// ============== Bar chart for focus node ==============
const barChartOption = computed(() => {
  const data = focusNeighbors.value
  if (!data.length) {
    return { backgroundColor: 'transparent', series: [] }
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(10,14,26,0.95)',
      borderColor: 'rgba(34,211,238,0.3)',
      borderWidth: 1,
      textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: (p) => `Node_${p[0].name}<br/>权重: <span style="color:#f97316">${p[0].value.toFixed(4)}</span>`
    },
    grid: { left: 45, right: 10, top: 5, bottom: 20 },
    xAxis: {
      type: 'category',
      data: data.map(d => d.node),
      axisLine: { lineStyle: { color: '#1e293b' } },
      axisLabel: { color: '#64748b', fontSize: 9, rotate: 45 }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#1e293b' } },
      axisLabel: { color: '#475569', fontSize: 8, formatter: (v) => v.toFixed(3) }
    },
    series: [{
      type: 'bar',
      data: data.map((d, i) => ({
        value: d.weight,
        itemStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: i === 0 ? '#ef4444' : '#f97316' },
              { offset: 1, color: i === 0 ? '#ef444440' : '#f9731640' }
            ]
          },
          borderRadius: [3, 3, 0, 0]
        }
      })),
      barWidth: '60%'
    }]
  }
})

// ============== Interactions ==============
const handleHeatmapClick = (params) => {
  if (params.data) {
    const [x, y, weight] = params.data
    const sampleRate = 3
    selectPair(x * sampleRate, y * sampleRate, weight)
  }
}

const handleBarClick = (params) => {
  const nodeId = focusNeighbors.value[params.dataIndex]?.node
  if (nodeId !== undefined) {
    const w = focusNeighbors.value[params.dataIndex].weight
    selectPair(focusNode.value, nodeId, w)
  }
}

const selectPair = (source, target, weight) => {
  selectedPair.value = { source, target, weight }
  focusNode.value = source
  updateFocusNeighbors()
  emit('node-pair-click', source, target)
}

onMounted(() => {
  fetchAttentionMatrix()
})
</script>

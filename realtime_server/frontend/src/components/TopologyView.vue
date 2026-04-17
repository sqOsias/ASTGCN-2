<template>
  <div class="h-full flex bg-[#0a0e1a]">
    <!-- Main Graph Area -->
    <div class="flex-1 relative overflow-hidden">
      <!-- Top Control Bar -->
      <div class="absolute top-4 left-4 right-4 z-10 flex justify-between items-center">
        <!-- Left: Title & Mode -->
        <div class="flex items-center gap-4">
          <div class="text-cyan-400 font-bold text-lg tracking-wider">
            <span class="text-cyan-300">◈</span> 全局态势监控
          </div>
          <el-button-group size="small">
            <el-button 
              :type="viewMode === 'current' ? 'primary' : 'default'" 
              @click="viewMode = 'current'"
            >
              实时路况
            </el-button>
            <el-button 
              :type="viewMode === 'future' ? 'primary' : 'default'" 
              @click="viewMode = 'future'"
            >
              预测态势
            </el-button>
          </el-button-group>
        </div>
        
        <!-- Center: Replay Controls -->
        <div class="flex items-center gap-3 bg-slate-900/80 px-3 py-1.5 rounded-full border border-slate-700/50">
          <el-button 
            size="small" 
            circle 
            :type="isReplaying ? 'danger' : 'primary'"
            @click="toggleReplay"
          >
            <el-icon><VideoPlay v-if="!isReplaying" /><VideoPause v-else /></el-icon>
          </el-button>
          <el-slider 
            v-model="replayIndex" 
            :min="0" 
            :max="Math.max(0, historyLength - 1)"
            :disabled="historyLength === 0"
            :show-tooltip="false"
            style="width: 120px"
            size="small"
          />
          <span class="text-xs text-slate-400 font-mono w-16">
            {{ replayTimeLabel }}
          </span>
        </div>
        
        <!-- Right: Focus Mode Toggle -->
        <div class="flex items-center gap-3">
          <div class="text-xs text-slate-400">震中聚焦</div>
          <el-switch 
            v-model="epicenterMode" 
            active-color="#ef4444"
            inactive-color="#334155"
            size="small"
          />
        </div>
      </div>
      
      <!-- Breathing Legend -->
      <div class="absolute bottom-4 left-4 z-10">
        <div class="flex items-center gap-6 text-xs">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span class="text-cyan-400/70">畅通 &gt;50</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
            <span class="text-amber-400/70">缓行 20-50</span>
          </div>
          <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.8)] animate-pulse"></span>
            <span class="text-red-400">拥堵 &lt;20</span>
          </div>
        </div>
      </div>
      
      <!-- Epicenter Mode Indicator -->
      <div v-if="epicenterMode" class="absolute top-16 left-1/2 -translate-x-1/2 z-10">
        <div class="px-4 py-1.5 bg-red-500/20 border border-red-500/50 rounded-full text-red-400 text-xs flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
          震中聚焦模式 · 显示 TOP {{ epicenterNodes.size }} 拥堵核心
        </div>
      </div>
      
      <!-- ECharts Graph -->
      <v-chart 
        ref="chartRef"
        class="w-full h-full" 
        :option="chartOption" 
        :autoresize="true"
        @click="handleChartClick"
      />
      
      <!-- Scan Line Effect -->
      <div class="absolute inset-0 pointer-events-none overflow-hidden">
        <div class="scan-line"></div>
      </div>
    </div>
    
    <!-- Right Sidebar -->
    <div class="w-72 border-l border-cyan-500/10 flex flex-col bg-[#0d1220]">
      <!-- Metrics Panel -->
      <div class="p-4 border-b border-cyan-500/10">
        <div class="text-xs text-slate-500 mb-3 flex items-center gap-2">
          <span class="w-1 h-3 bg-cyan-500 rounded"></span>
          实时误差回溯
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div class="bg-slate-900/50 rounded-lg p-3 text-center">
            <div class="text-2xl font-mono font-bold text-cyan-400">{{ metrics.mae.toFixed(2) }}</div>
            <div class="text-[10px] text-slate-500 mt-1">MAE (km/h)</div>
          </div>
          <div class="bg-slate-900/50 rounded-lg p-3 text-center">
            <div class="text-2xl font-mono font-bold text-purple-400">{{ metrics.rmse.toFixed(2) }}</div>
            <div class="text-[10px] text-slate-500 mt-1">RMSE (km/h)</div>
          </div>
        </div>
        <!-- Accuracy Bar -->
        <div class="mt-3">
          <div class="flex justify-between text-[10px] text-slate-500 mb-1">
            <span>精度评级</span>
            <span :class="accuracyTextColor">{{ accuracyLabel }}</span>
          </div>
          <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div 
              class="h-full transition-all duration-700 rounded-full"
              :class="accuracyColor"
              :style="{ width: accuracyPercent + '%' }"
            ></div>
          </div>
        </div>
      </div>
      
      <!-- Congestion Forecast -->
      <div class="flex-1 flex flex-col min-h-0 p-4">
        <div class="text-xs text-slate-500 mb-3 flex items-center gap-2">
          <span class="w-1 h-3 bg-red-500 rounded"></span>
          未来1h拥堵预警
        </div>
        <div class="flex-1 overflow-auto space-y-2 scrollbar-thin">
          <div 
            v-for="(node, index) in topCongested" 
            :key="node.node_id"
            class="group flex items-center gap-2 p-2 rounded-lg bg-slate-900/30 hover:bg-red-500/10 cursor-pointer transition-all duration-300 border border-transparent hover:border-red-500/30"
            @click="focusOnNode(node.node_id)"
          >
            <div 
              class="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold shrink-0"
              :class="index === 0 ? 'bg-red-500 text-white' : 'bg-slate-800 text-slate-400'"
            >
              {{ index + 1 }}
            </div>
            <div class="flex-1 min-w-0">
              <div class="text-xs font-medium text-slate-300 truncate">传感器 #{{ node.node_id }}</div>
              <div class="text-[10px] text-slate-500">当前 {{ node.current_speed?.toFixed(0) }} km/h</div>
            </div>
            <div class="text-right shrink-0">
              <div class="text-sm font-mono font-bold" :class="getSpeedTextColor(node.avg_speed)">{{ node.avg_speed?.toFixed(0) }}</div>
              <div class="text-[10px] text-slate-600">预测</div>
            </div>
            <div class="w-1.5 h-8 bg-slate-800 rounded-full overflow-hidden shrink-0">
              <div 
                class="w-full bg-gradient-to-t from-red-500 to-red-400 rounded-full transition-all"
                :style="{ height: Math.min(100, (1 - node.avg_speed / 60) * 100) + '%' }"
              ></div>
            </div>
          </div>
          
          <div v-if="topCongested.length === 0" class="text-center text-slate-600 py-8">
            <div class="text-2xl mb-2">◌</div>
            <div class="text-xs">等待数据...</div>
          </div>
        </div>
      </div>
      
      <!-- Network Overview -->
      <div class="p-4 border-t border-cyan-500/10">
        <div class="text-xs text-slate-500 mb-3 flex items-center gap-2">
          <span class="w-1 h-3 bg-emerald-500 rounded"></span>
          全网态势
        </div>
        <div class="flex justify-between items-end">
          <div class="text-center">
            <div class="text-xl font-bold text-emerald-400">{{ networkStats.smooth }}</div>
            <div class="text-[10px] text-slate-500">畅通</div>
          </div>
          <div class="text-center">
            <div class="text-xl font-bold text-amber-400">{{ networkStats.slow }}</div>
            <div class="text-[10px] text-slate-500">缓行</div>
          </div>
          <div class="text-center">
            <div class="text-xl font-bold text-red-400">{{ networkStats.congested }}</div>
            <div class="text-[10px] text-slate-500">拥堵</div>
          </div>
          <div class="text-center">
            <div class="text-xl font-bold text-slate-400">{{ totalNodes }}</div>
            <div class="text-[10px] text-slate-500">总计</div>
          </div>
        </div>
        <!-- Mini Progress -->
        <div class="mt-3 h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
          <div class="bg-emerald-500 transition-all" :style="{ width: (networkStats.smooth / totalNodes * 100) + '%' }"></div>
          <div class="bg-amber-500 transition-all" :style="{ width: (networkStats.slow / totalNodes * 100) + '%' }"></div>
          <div class="bg-red-500 transition-all" :style="{ width: (networkStats.congested / totalNodes * 100) + '%' }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, shallowRef } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { GraphChart, EffectScatterChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import VChart from 'vue-echarts'

use([CanvasRenderer, GraphChart, EffectScatterChart, TooltipComponent, LegendComponent])

const props = defineProps({
  networkData: { type: Array, default: () => [] },
  topology: { type: Object, default: () => ({ nodes: [], edges: [] }) },
  metrics: { type: Object, default: () => ({ mae: 0, rmse: 0 }) },
  historyBuffer: { type: Array, default: () => [] }
})

const emit = defineEmits(['node-click'])

const chartRef = shallowRef(null)
const viewMode = ref('current')
const epicenterMode = ref(false)
const totalNodes = 307

// Replay state
const isReplaying = ref(false)
const replayIndex = ref(0)
let replayTimer = null

const historyLength = computed(() => props.historyBuffer.length)

const replayTimeLabel = computed(() => {
  if (historyLength.value === 0) return '--:--'
  const offset = replayIndex.value - (historyLength.value - 1)
  return offset === 0 ? 'NOW' : `${offset * 5}m`
})

const toggleReplay = () => {
  if (isReplaying.value) {
    clearInterval(replayTimer)
    isReplaying.value = false
  } else {
    isReplaying.value = true
    replayIndex.value = 0
    replayTimer = setInterval(() => {
      if (replayIndex.value < historyLength.value - 1) {
        replayIndex.value++
      } else {
        clearInterval(replayTimer)
        isReplaying.value = false
      }
    }, 500)
  }
}

// Current display data (live or replay)
const displayData = computed(() => {
  if (isReplaying.value || replayIndex.value < historyLength.value - 1) {
    return props.historyBuffer[replayIndex.value]?.data || []
  }
  return props.networkData
})

// Color helpers
const getSpeedColor = (speed) => {
  if (speed > 50) return 'rgba(34, 211, 238, 0.8)' // Cyan
  if (speed > 20) return 'rgba(251, 191, 36, 0.9)' // Amber
  return 'rgba(239, 68, 68, 1)' // Red
}

const getSpeedTextColor = (speed) => {
  if (speed > 50) return 'text-cyan-400'
  if (speed > 20) return 'text-amber-400'
  return 'text-red-400'
}

// Node speeds (uses displayData for replay support)
const nodeSpeedMap = computed(() => {
  const map = {}
  const data = displayData.value
  if (!data.length) return map
  data.forEach(node => {
    map[node.node_id] = viewMode.value === 'current' 
      ? node.current_real_speed 
      : (node.future_pred_speeds?.[5] || node.current_real_speed)
  })
  return map
})

// Network stats
const networkStats = computed(() => {
  const speeds = Object.values(nodeSpeedMap.value)
  return {
    smooth: speeds.filter(s => s > 50).length,
    slow: speeds.filter(s => s > 20 && s <= 50).length,
    congested: speeds.filter(s => s <= 20).length
  }
})

// Top congested nodes
const topCongested = computed(() => {
  if (!props.networkData.length) return []
  return props.networkData
    .map(node => ({
      node_id: node.node_id,
      current_speed: node.current_real_speed,
      avg_speed: node.future_pred_speeds 
        ? node.future_pred_speeds.reduce((a, b) => a + b, 0) / node.future_pred_speeds.length 
        : node.current_real_speed
    }))
    .sort((a, b) => a.avg_speed - b.avg_speed)
    .slice(0, 8)
})

// Epicenter nodes (Top 5 + their neighbors)
const epicenterNodes = computed(() => {
  const nodes = new Set()
  const top5 = topCongested.value.slice(0, 5).map(n => n.node_id)
  top5.forEach(id => nodes.add(id))
  
  // Add 1-hop neighbors
  props.topology.edges.forEach(edge => {
    if (top5.includes(edge.source)) nodes.add(edge.target)
    if (top5.includes(edge.target)) nodes.add(edge.source)
  })
  return nodes
})

// Accuracy
const accuracyPercent = computed(() => {
  const mae = props.metrics.mae
  return Math.max(0, Math.min(100, (1 - mae / 10) * 100))
})

const accuracyColor = computed(() => {
  const p = accuracyPercent.value
  if (p > 70) return 'bg-gradient-to-r from-emerald-600 to-emerald-400'
  if (p > 40) return 'bg-gradient-to-r from-amber-600 to-amber-400'
  return 'bg-gradient-to-r from-red-600 to-red-400'
})

const accuracyTextColor = computed(() => {
  const p = accuracyPercent.value
  if (p > 70) return 'text-emerald-400'
  if (p > 40) return 'text-amber-400'
  return 'text-red-400'
})

const accuracyLabel = computed(() => {
  const p = accuracyPercent.value
  if (p > 70) return '优秀'
  if (p > 40) return '良好'
  return '需改进'
})

// Generate uniform grid positions for starfield mode
const generateGridPositions = () => {
  const positions = {}
  const cols = 20
  const rows = Math.ceil(totalNodes / cols)
  const width = 700
  const height = 550
  const cellW = width / cols
  const cellH = height / rows
  const offsetX = 80
  const offsetY = 80
  
  for (let i = 0; i < totalNodes; i++) {
    const col = i % cols
    const row = Math.floor(i / cols)
    // Add slight randomness for organic feel
    positions[i] = {
      x: offsetX + col * cellW + (Math.random() - 0.5) * cellW * 0.3,
      y: offsetY + row * cellH + (Math.random() - 0.5) * cellH * 0.3
    }
  }
  return positions
}

const gridPositions = generateGridPositions()

// ECharts option
const chartOption = computed(() => {
  const isEpicenter = epicenterMode.value
  const epicenterSet = epicenterNodes.value
  const top5Ids = topCongested.value.slice(0, 5).map(n => n.node_id)
  
  // Build nodes
  const nodes = []
  for (let i = 0; i < totalNodes; i++) {
    const speed = nodeSpeedMap.value[i] || 60
    const pos = gridPositions[i] || { x: 400, y: 300 }
    const isCongested = speed < 20
    const isTop5 = top5Ids.includes(i)
    const inEpicenter = epicenterSet.has(i)
    
    // In epicenter mode, hide non-epicenter nodes
    if (isEpicenter && !inEpicenter) continue
    
    nodes.push({
      id: String(i),
      name: `传感器 #${i}`,
      x: pos.x,
      y: pos.y,
      fixed: true,
      symbolSize: isTop5 ? 16 : isCongested ? 10 : 5,
      itemStyle: {
        color: isCongested ? '#ef4444' : speed < 50 ? 'rgba(251,191,36,0.7)' : 'rgba(34,211,238,0.4)',
        borderColor: isTop5 ? '#fef2f2' : 'transparent',
        borderWidth: isTop5 ? 2 : 0,
        shadowColor: isCongested ? 'rgba(239,68,68,0.8)' : 'transparent',
        shadowBlur: isCongested ? 15 : 0
      },
      value: speed
    })
  }
  
  // Build edges - color based on congestion level
  const edges = []
  props.topology.edges.forEach(edge => {
    const sourceSpeed = nodeSpeedMap.value[edge.source] || 60
    const targetSpeed = nodeSpeedMap.value[edge.target] || 60
    const avgSpeed = (sourceSpeed + targetSpeed) / 2
    
    // In epicenter mode, only show edges within epicenter
    if (isEpicenter) {
      if (!epicenterSet.has(edge.source) || !epicenterSet.has(edge.target)) return
    }
    
    // Determine edge style based on speed
    let edgeColor, edgeWidth, shadowColor, shadowBlur
    if (avgSpeed < 40) {
      // Severe congestion - bright red with glow
      edgeColor = 'rgba(239, 68, 68, 0.95)'
      edgeWidth = 3
      shadowColor = 'rgba(239, 68, 68, 0.8)'
      shadowBlur = 12
    } else if (avgSpeed < 80) {
      // Slow traffic - yellow/amber
      edgeColor = 'rgba(251, 191, 36, 0.8)'
      edgeWidth = 2
      shadowColor = 'rgba(251, 191, 36, 0.4)'
      shadowBlur = 6
    } else {
      // Normal flow - subtle cyan (hidden by default, show on hover)
      edgeColor = 'rgba(34, 211, 238, 0.15)'
      edgeWidth = 0.5
      shadowColor = 'transparent'
      shadowBlur = 0
    }
    
    edges.push({
      source: String(edge.source),
      target: String(edge.target),
      lineStyle: {
        color: edgeColor,
        width: edgeWidth,
        shadowColor: shadowColor,
        shadowBlur: shadowBlur
      }
    })
  })
  
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(10, 14, 26, 0.95)',
      borderColor: 'rgba(34, 211, 238, 0.3)',
      borderWidth: 1,
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter: (params) => {
        if (params.dataType === 'node') {
          const speed = params.value
          const status = speed > 50 ? '畅通' : speed > 20 ? '缓行' : '拥堵'
          const statusColor = speed > 50 ? '#22d3ee' : speed > 20 ? '#fbbf24' : '#ef4444'
          return `<div style="font-weight:600;margin-bottom:4px">${params.name}</div>
                  <div style="color:#94a3b8">车速: <span style="color:${statusColor};font-weight:600">${speed?.toFixed(1)} km/h</span></div>
                  <div style="color:#94a3b8">状态: <span style="color:${statusColor}">${status}</span></div>`
        }
        return ''
      }
    },
    series: [{
      type: 'graph',
      layout: 'none',
      animation: true,
      animationDuration: 500,
      data: nodes,
      links: edges,
      roam: true,
      zoom: isEpicenter ? 1.8 : 1.1,
      center: isEpicenter && top5Ids.length > 0 ? [gridPositions[top5Ids[0]]?.x || 400, gridPositions[top5Ids[0]]?.y || 300] : undefined,
      emphasis: {
        focus: 'adjacency',
        lineStyle: { width: 4 },
        itemStyle: { borderColor: '#fff', borderWidth: 2 }
      },
      blur: {
        itemStyle: { opacity: 0.3 }
      },
      label: { show: false },
      lineStyle: {
        curveness: 0.2
      }
    }]
  }
})

const handleChartClick = (params) => {
  if (params.dataType === 'node') {
    const nodeId = parseInt(params.data.id)
    emit('node-click', nodeId)
  }
}

const focusOnNode = (nodeId) => {
  emit('node-click', nodeId)
}
</script>

<style scoped>
.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.3), transparent);
  animation: scan 4s linear infinite;
}

@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

.scrollbar-thin::-webkit-scrollbar {
  width: 4px;
}

.scrollbar-thin::-webkit-scrollbar-track {
  background: rgba(30, 41, 59, 0.5);
}

.scrollbar-thin::-webkit-scrollbar-thumb {
  background: rgba(71, 85, 105, 0.5);
  border-radius: 2px;
}

.scrollbar-thin::-webkit-scrollbar-thumb:hover {
  background: rgba(100, 116, 139, 0.5);
}
</style>

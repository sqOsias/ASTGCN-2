<template>
  <Splitpanes class="h-full bg-[#0a0e1a]">
    <Pane :size="75" :min-size="30">
    <!-- Main Graph Area -->
    <div class="h-full relative overflow-hidden">
      <!-- Top Bar -->
      <div class="absolute top-4 left-4 right-4 z-10 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <div class="text-cyan-400 font-bold text-lg tracking-wider">
            <span class="text-cyan-300">◈</span> 智能路径规划
          </div>
          <div v-if="planningActive" class="flex items-center gap-2">
            <div class="px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-cyan-400 text-xs flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
              导航诱导模式
            </div>
            <div class="px-3 py-1 bg-amber-500/10 border border-amber-500/30 rounded-full text-amber-400 text-xs">
              ‖ 仿真已暂停
            </div>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <el-button v-if="planningActive" size="small" type="danger" plain @click="resetPlanning">
            退出规划
          </el-button>
        </div>
      </div>

      <!-- Legend -->
      <div class="absolute bottom-4 left-4 z-10">
        <div v-if="planningActive && routes.length" class="flex items-center gap-6 text-xs">
          <div v-for="(route, idx) in routes" :key="idx" class="flex items-center gap-2 cursor-pointer" @click="activeRoute = idx">
            <span class="w-3 h-3 rounded-full" :style="{ background: routeColors[idx], boxShadow: activeRoute === idx ? `0 0 10px ${routeColors[idx]}` : 'none' }"></span>
            <span :style="{ color: routeColors[idx] }">{{ routeLabels[idx] }}</span>
          </div>
        </div>
        <div v-else class="flex items-center gap-6 text-xs">
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

      <!-- ECharts Graph -->
      <v-chart
        ref="chartRef"
        class="w-full h-full"
        :option="chartOption"
        :autoresize="true"
        @click="handleChartClick"
      />

      <!-- Scan line effect -->
      <div class="absolute inset-0 pointer-events-none overflow-hidden">
        <div class="scan-line"></div>
      </div>
    </div>

    </Pane>
    <Pane :size="25" :min-size="8">
    <!-- Right Sidebar -->
    <Splitpanes horizontal class="h-full bg-[#0d1220]">
      <Pane :size="40" :min-size="15">
      <!-- Route Input Panel -->
      <div class="h-full p-4 overflow-auto">
        <div class="text-xs text-slate-500 mb-3 flex items-center gap-2">
          <span class="w-1 h-3 bg-cyan-500 rounded"></span>
          出行需求录入
        </div>
        <div class="space-y-3">
          <div>
            <label class="text-[10px] text-slate-500 mb-1 block">出发节点</label>
            <el-select v-model="sourceNode" filterable placeholder="选择或点击地图" size="small" class="w-full" @change="clearResults">
              <el-option-group v-for="comp in componentGroups" :key="comp.label" :label="comp.label">
                <el-option v-for="n in comp.nodes" :key="n" :label="`#${n}`" :value="n" />
              </el-option-group>
            </el-select>
          </div>
          <div>
            <label class="text-[10px] text-slate-500 mb-1 block">目的节点</label>
            <el-select v-model="targetNode" filterable placeholder="选择或点击地图" size="small" class="w-full" @change="clearResults">
              <el-option-group v-for="comp in componentGroups" :key="comp.label" :label="comp.label">
                <el-option v-for="n in comp.nodes" :key="n" :label="`#${n}`" :value="n" />
              </el-option-group>
            </el-select>
          </div>
          <div class="flex items-center gap-2">
            <el-button type="primary" class="flex-1" size="small" :loading="loading" @click="planRoute" :disabled="sourceNode === null || targetNode === null">
              <el-icon class="mr-1"><Position /></el-icon>
              开始规划
            </el-button>
            <el-button size="small" @click="swapNodes" :disabled="sourceNode === null && targetNode === null">
              <el-icon><Sort /></el-icon>
            </el-button>
          </div>
          <div v-if="selectMode" class="text-xs text-cyan-400 animate-pulse text-center">
            点击地图选择{{ selectMode === 'source' ? '出发' : '目的' }}节点
          </div>
          <div class="flex gap-2">
            <el-button size="small" text type="primary" @click="startSelectMode('source')">
              地图选起点
            </el-button>
            <el-button size="small" text type="warning" @click="startSelectMode('target')">
              地图选终点
            </el-button>
          </div>
          <div v-if="errorMsg" class="text-xs text-red-400 bg-red-500/10 p-2 rounded">{{ errorMsg }}</div>
        </div>
      </div>
      </Pane>
      <Pane :size="60" :min-size="15">
      <!-- Route Results -->
      <div class="h-full p-4 overflow-auto scrollbar-thin">
        <div v-if="routes.length" class="space-y-3">
          <div class="text-xs text-slate-500 mb-1 flex items-center gap-2">
            <span class="w-1 h-3 bg-emerald-500 rounded"></span>
            规划结果 ({{ routes.length }} 条路线)
          </div>
          <div
            v-for="(route, idx) in routes"
            :key="idx"
            class="route-card p-3 rounded-lg border cursor-pointer transition-all duration-300"
            :class="activeRoute === idx
              ? 'border-current bg-opacity-20 shadow-lg'
              : 'border-slate-700/50 bg-slate-900/30 hover:border-slate-600'"
            :style="activeRoute === idx ? { borderColor: routeColors[idx], background: routeColors[idx] + '15', boxShadow: `0 0 20px ${routeColors[idx]}33` } : {}"
            @click="activeRoute = idx"
          >
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full" :style="{ background: routeColors[idx] }"></span>
                <span class="text-sm font-bold" :style="{ color: routeColors[idx] }">{{ routeLabels[idx] }}</span>
              </div>
              <span v-if="idx === 0" class="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-400">推荐</span>
            </div>
            <div class="grid grid-cols-3 gap-2 text-center">
              <div>
                <div class="text-lg font-mono font-bold text-white">{{ route.eta_minutes }}</div>
                <div class="text-[10px] text-slate-500">预计(分钟)</div>
              </div>
              <div>
                <div class="text-lg font-mono font-bold text-slate-300">{{ route.total_distance_km }}</div>
                <div class="text-[10px] text-slate-500">里程(km)</div>
              </div>
              <div>
                <div class="text-lg font-mono font-bold text-slate-300">{{ route.path.length }}</div>
                <div class="text-[10px] text-slate-500">途径节点</div>
              </div>
            </div>
            <!-- Speed Profile Mini Chart -->
            <div class="mt-2 h-16 bg-slate-900/60 rounded overflow-hidden">
              <v-chart :option="speedChartOption(route, idx)" :autoresize="true" class="w-full h-full" />
            </div>
          </div>

          <!-- Static ETA comparison -->
          <div v-if="staticEta !== null" class="mt-2 p-3 rounded-lg bg-slate-900/30 border border-slate-700/30">
            <div class="text-xs text-slate-500 mb-1">对比：仅用当前速度规划</div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-400">静态ETA</span>
              <span class="text-lg font-mono font-bold text-slate-500">{{ staticEta }} 分钟</span>
            </div>
            <div v-if="routes.length && routes[0].eta_minutes !== staticEta" class="text-[10px] mt-1" :class="routes[0].eta_minutes < staticEta ? 'text-emerald-400' : 'text-amber-400'">
              {{ routes[0].eta_minutes < staticEta ? '▼ 动态规划节省' : '▲ 动态规划多出' }}
              {{ Math.abs(routes[0].eta_minutes - staticEta).toFixed(1) }} 分钟
            </div>
          </div>

          <!-- Plan note -->
          <div v-if="planNote" class="p-2 rounded-lg text-xs" :class="predictionCoverage > 0 ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'">
            {{ planNote }}
          </div>

          <!-- Route info -->
          <div class="text-[10px] text-slate-600 leading-relaxed">
            <p>● 预测时间窗口: {{ predictionHorizon }} 分钟</p>
            <p>● 路段速度 = 两端节点预测速度均值</p>
            <p>● 超出预测窗口自动降级为末步速度</p>
            <p>● 规划时仿真自动暂停，退出后恢复</p>
          </div>
        </div>
        <div v-else-if="!loading" class="flex-1 flex items-center justify-center">
          <div class="text-center text-slate-600">
            <div class="text-4xl mb-3">🗺️</div>
            <div class="text-sm">选择起终点开始规划</div>
            <div class="text-[10px] mt-1 text-slate-700">基于ASTGCN预测的前瞻性寻路</div>
          </div>
        </div>
      </div>
      </Pane>
    </Splitpanes>
    </Pane>
  </Splitpanes>
</template>

<script setup>
import { ref, reactive, computed, watch, shallowRef, onUnmounted } from 'vue'
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { GraphChart, EffectScatterChart, LineChart } from 'echarts/charts'
import { TooltipComponent, GridComponent } from 'echarts/components'
import VChart from 'vue-echarts'

use([CanvasRenderer, GraphChart, EffectScatterChart, LineChart, TooltipComponent, GridComponent])

const props = defineProps({
  networkData: { type: Array, default: () => [] },
  topology: { type: Object, default: () => ({ nodes: [], edges: [] }) },
  metrics: { type: Object, default: () => ({ mae: 0, rmse: 0 }) }
})

const chartRef = shallowRef(null)
const sourceNode = ref(null)
const targetNode = ref(null)
const loading = ref(false)
const errorMsg = ref('')
const selectMode = ref(null) // 'source' | 'target' | null
const planningActive = ref(false)
const activeRoute = ref(0)
const routes = ref([])
const staticEta = ref(null)
const predictionHorizon = ref(60)
const predictionCoverage = ref(0)
const planNote = ref('')
const components = ref([])
const animatingRoute = ref(-1) // which route is animating
const animatedEdges = ref(new Set()) // edges that have been animated
const wasPaused = ref(false) // track if simulation was already paused before planning

const routeColors = ['#00f2ff', '#ff8c00', '#c050ff']
const routeLabels = ['最优推荐路线', '备选路线 A', '备选路线 B']
const totalNodes = computed(() => props.topology.nodes?.length || 0)

let animTimers = []

// ============== Component groups for select dropdowns ==============
const componentGroups = computed(() => {
  if (components.value.length === 0) {
    // Fallback: single group with all nodes
    const all = []
    for (let i = 0; i < totalNodes.value; i++) all.push(i)
    return [{ label: '全部节点', nodes: all }]
  }
  return components.value.map(c => ({
    label: `分量 ${c.id} (${c.size} 节点)`,
    nodes: c.nodes.sort((a, b) => a - b)
  }))
})

// ============== Fetch components on mount ==============
const fetchComponents = async () => {
  try {
    const res = await fetch('/api/route/components')
    const data = await res.json()
    components.value = data.components
  } catch (e) {
    console.error('Failed to fetch components:', e)
  }
}
fetchComponents()

// ============== Node positions from backend topology ==============
const nodePositions = computed(() => {
  const positions = {}
  if (props.topology.nodes && props.topology.nodes.length) {
    props.topology.nodes.forEach(n => {
      positions[n.id] = { x: n.x, y: n.y }
    })
  }
  return positions
})

// ============== Speed map ==============
const nodeSpeedMap = computed(() => {
  const map = {}
  props.networkData.forEach(n => { map[n.node_id] = n.current_real_speed })
  return map
})

// ============== Edge set for quick lookup ==============
const edgeSet = computed(() => {
  const s = new Set()
  props.topology.edges.forEach(e => {
    s.add(`${e.source}-${e.target}`)
    s.add(`${e.target}-${e.source}`)
  })
  return s
})

// ============== Route edge sets for highlighting ==============
const routeEdgeSets = computed(() => {
  return routes.value.map(r => {
    const s = new Set()
    for (let i = 0; i < r.path.length - 1; i++) {
      s.add(`${r.path[i]}-${r.path[i+1]}`)
      s.add(`${r.path[i+1]}-${r.path[i]}`)
    }
    return s
  })
})

const routeNodeSets = computed(() => {
  return routes.value.map(r => new Set(r.path))
})

// ============== Actions ==============
const swapNodes = () => {
  const tmp = sourceNode.value
  sourceNode.value = targetNode.value
  targetNode.value = tmp
}

const clearResults = () => {
  // Don't clear when just changing selection
}

const pauseSimulation = async () => {
  try {
    const res = await fetch('/api/stats')
    const stats = await res.json()
    // Check if simulation is already paused (simulation_speed context)
    wasPaused.value = false // we'll track via the running state
    await fetch('/api/simulation/pause', { method: 'POST' })
  } catch (e) { console.error('Failed to pause:', e) }
}

const resumeSimulation = async () => {
  if (!wasPaused.value) {
    try { await fetch('/api/simulation/resume', { method: 'POST' }) }
    catch (e) { console.error('Failed to resume:', e) }
  }
}

const resetPlanning = () => {
  planningActive.value = false
  routes.value = []
  staticEta.value = null
  predictionCoverage.value = 0
  planNote.value = ''
  activeRoute.value = 0
  animatingRoute.value = -1
  animatedEdges.value = new Set()
  animTimers.forEach(t => clearTimeout(t))
  animTimers = []
  // Resume simulation when exiting planning
  resumeSimulation()
}

const startSelectMode = (mode) => {
  selectMode.value = mode
}

const handleChartClick = (params) => {
  if (params.dataType === 'node') {
    const nodeId = parseInt(params.data.id)
    if (selectMode.value === 'source') {
      sourceNode.value = nodeId
      selectMode.value = null
    } else if (selectMode.value === 'target') {
      targetNode.value = nodeId
      selectMode.value = null
    }
  }
}

const planRoute = async () => {
  if (sourceNode.value === null || targetNode.value === null) return
  if (sourceNode.value === targetNode.value) {
    errorMsg.value = '起终点不能相同'
    return
  }

  loading.value = true
  errorMsg.value = ''
  // Reset without resuming simulation
  planningActive.value = false
  routes.value = []
  animTimers.forEach(t => clearTimeout(t))
  animTimers = []

  // Pause simulation to freeze prediction data
  await pauseSimulation()

  try {
    const res = await fetch(`/api/route/plan?source=${sourceNode.value}&target=${targetNode.value}&k=3`)
    const data = await res.json()

    if (data.error) {
      errorMsg.value = data.error
      resumeSimulation()
      return
    }

    routes.value = data.routes || []
    staticEta.value = data.static_eta_minutes
    predictionHorizon.value = data.prediction_horizon_min || 60
    predictionCoverage.value = data.prediction_coverage || 0
    planNote.value = data.note || ''

    if (routes.value.length > 0) {
      planningActive.value = true
      activeRoute.value = 0
      startRouteAnimation()
    } else {
      resumeSimulation()
    }
  } catch (e) {
    errorMsg.value = '请求失败: ' + e.message
    resumeSimulation()
  } finally {
    loading.value = false
  }
}

// ============== Sequential Route Animation ==============
const startRouteAnimation = () => {
  animTimers.forEach(t => clearTimeout(t))
  animTimers = []
  animatedEdges.value = new Set()
  animatingRoute.value = 0

  let delay = 0
  routes.value.forEach((route, rIdx) => {
    const path = route.path
    // Animate each edge one by one
    for (let i = 0; i < path.length - 1; i++) {
      const timer = setTimeout(() => {
        animatingRoute.value = rIdx
        animatedEdges.value = new Set([...animatedEdges.value, `${rIdx}:${path[i]}-${path[i+1]}`])
      }, delay)
      animTimers.push(timer)
      delay += 80  // 80ms per edge segment
    }
    // After this route finishes, brief pause
    delay += 400
  })
  // Animation complete
  const timer = setTimeout(() => {
    animatingRoute.value = -1
  }, delay)
  animTimers.push(timer)
}

// ============== Speed Profile Mini Chart ==============
const speedChartOption = (route, idx) => {
  const profile = route.speed_profile || []
  return {
    backgroundColor: 'transparent',
    grid: { left: 0, right: 0, top: 2, bottom: 0, containLabel: false },
    xAxis: { type: 'category', show: false, data: profile.map(p => p.node_id) },
    yAxis: { type: 'value', show: false, min: 0, max: 140 },
    series: [{
      type: 'line',
      data: profile.map(p => p.speed_kmh),
      smooth: true,
      symbol: 'none',
      lineStyle: { color: routeColors[idx], width: 1.5 },
      areaStyle: {
        color: {
          type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: routeColors[idx] + '40' },
            { offset: 1, color: routeColors[idx] + '05' }
          ]
        }
      }
    }]
  }
}

// ============== Main ECharts Option ==============
const chartOption = computed(() => {
  const isPlanning = planningActive.value
  const currentRouteEdges = routeEdgeSets.value
  const currentRouteNodes = routeNodeSets.value
  const aRoute = activeRoute.value

  // Build nodes
  const nodes = []
  for (let i = 0; i < totalNodes.value; i++) {
    const speed = nodeSpeedMap.value[i] || 60
    const isSource = i === sourceNode.value
    const isTarget = i === targetNode.value

    // Check if node is on any route
    let onRoute = -1
    if (isPlanning) {
      for (let r = 0; r < currentRouteNodes.length; r++) {
        if (currentRouteNodes[r].has(i)) { onRoute = r; break }
      }
    }

    const dimmed = isPlanning && onRoute === -1 && !isSource && !isTarget

    let color, size, borderColor, borderWidth, shadowColor, shadowBlur
    if (isSource) {
      color = '#00ff88'
      size = 18
      borderColor = '#ffffff'
      borderWidth = 3
      shadowColor = 'rgba(0,255,136,0.8)'
      shadowBlur = 20
    } else if (isTarget) {
      color = '#ff4466'
      size = 18
      borderColor = '#ffffff'
      borderWidth = 3
      shadowColor = 'rgba(255,68,102,0.8)'
      shadowBlur = 20
    } else if (isPlanning && onRoute >= 0) {
      color = routeColors[onRoute] || '#00f2ff'
      size = onRoute === aRoute ? 8 : 5
      borderColor = 'transparent'
      borderWidth = 0
      shadowColor = onRoute === aRoute ? routeColors[onRoute] + '80' : 'transparent'
      shadowBlur = onRoute === aRoute ? 10 : 0
    } else {
      const speedColor = speed > 50 ? 'rgba(34,211,238,0.4)' : speed > 20 ? 'rgba(251,191,36,0.7)' : 'rgba(239,68,68,1)'
      color = dimmed ? 'rgba(100,116,139,0.08)' : speedColor
      size = dimmed ? 2 : (speed < 20 ? 10 : 5)
      borderColor = 'transparent'
      borderWidth = 0
      shadowColor = (!dimmed && speed < 20) ? 'rgba(239,68,68,0.8)' : 'transparent'
      shadowBlur = (!dimmed && speed < 20) ? 15 : 0
    }

    const pos = nodePositions.value[i] || { x: 400, y: 300 }
    nodes.push({
      id: String(i),
      name: `传感器 #${i}`,
      x: pos.x,
      y: pos.y,
      fixed: true,
      symbolSize: size,
      itemStyle: { color, borderColor, borderWidth, shadowColor, shadowBlur },
      value: speed
    })
  }

  // Build edges
  const edges = []
  props.topology.edges.forEach(edge => {
    const key1 = `${edge.source}-${edge.target}`
    const sourceSpeed = nodeSpeedMap.value[edge.source] || 60
    const targetSpeed = nodeSpeedMap.value[edge.target] || 60
    const avgSpeed = (sourceSpeed + targetSpeed) / 2

    // Check if edge is on any route
    let onRoute = -1
    if (isPlanning) {
      for (let r = 0; r < currentRouteEdges.length; r++) {
        if (currentRouteEdges[r].has(key1)) { onRoute = r; break }
      }
    }

    const dimmed = isPlanning && onRoute === -1

    let edgeColor, edgeWidth, eShadowColor, eShadowBlur
    if (isPlanning && onRoute >= 0) {
      // Check animation state
      const animKey = `${onRoute}:${edge.source}-${edge.target}`
      const animKeyRev = `${onRoute}:${edge.target}-${edge.source}`
      const isAnimated = animatedEdges.value.has(animKey) || animatedEdges.value.has(animKeyRev)
      const isActiveRoute = onRoute === aRoute

      if (isAnimated || animatingRoute.value === -1) {
        edgeColor = routeColors[onRoute]
        edgeWidth = isActiveRoute ? 4 : 2.5
        eShadowColor = routeColors[onRoute] + (isActiveRoute ? 'AA' : '55')
        eShadowBlur = isActiveRoute ? 15 : 6
      } else {
        edgeColor = routeColors[onRoute] + '30'
        edgeWidth = 1
        eShadowColor = 'transparent'
        eShadowBlur = 0
      }
    } else if (dimmed) {
      edgeColor = 'rgba(100,116,139,0.04)'
      edgeWidth = 0.3
      eShadowColor = 'transparent'
      eShadowBlur = 0
    } else {
      if (avgSpeed < 40) {
        edgeColor = 'rgba(239,68,68,0.95)'
        edgeWidth = 3
        eShadowColor = 'rgba(239,68,68,0.8)'
        eShadowBlur = 12
      } else if (avgSpeed < 80) {
        edgeColor = 'rgba(251,191,36,0.8)'
        edgeWidth = 2
        eShadowColor = 'rgba(251,191,36,0.4)'
        eShadowBlur = 6
      } else {
        edgeColor = 'rgba(34,211,238,0.15)'
        edgeWidth = 0.5
        eShadowColor = 'transparent'
        eShadowBlur = 0
      }
    }

    edges.push({
      source: String(edge.source),
      target: String(edge.target),
      lineStyle: { color: edgeColor, width: edgeWidth, shadowColor: eShadowColor, shadowBlur: eShadowBlur }
    })
  })

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(10,14,26,0.95)',
      borderColor: 'rgba(34,211,238,0.3)',
      borderWidth: 1,
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter: (params) => {
        if (params.dataType === 'node') {
          const speed = params.value
          const nodeId = parseInt(params.data.id)
          const tags = []
          if (nodeId === sourceNode.value) tags.push('<span style="color:#00ff88">● 起点</span>')
          if (nodeId === targetNode.value) tags.push('<span style="color:#ff4466">● 终点</span>')
          const status = speed > 50 ? '畅通' : speed > 20 ? '缓行' : '拥堵'
          const sc = speed > 50 ? '#22d3ee' : speed > 20 ? '#fbbf24' : '#ef4444'
          return `<div style="font-weight:600;margin-bottom:4px">${params.name} ${tags.join(' ')}</div>
                  <div style="color:#94a3b8">车速: <span style="color:${sc};font-weight:600">${speed?.toFixed(1)} km/h</span></div>
                  <div style="color:#94a3b8">状态: <span style="color:${sc}">${status}</span></div>`
        }
        return ''
      }
    },
    series: [{
      type: 'graph',
      layout: 'none',
      animation: false,
      data: nodes,
      links: edges,
      roam: true,
      zoom: 1.1,
      emphasis: {
        focus: 'adjacency',
        lineStyle: { width: 4 },
        itemStyle: { borderColor: '#fff', borderWidth: 2 }
      },
      blur: { itemStyle: { opacity: 0.3 } },
      label: { show: false },
      lineStyle: { curveness: 0.15, opacity: 0.7 }
    }]
  }
})

onUnmounted(() => {
  animTimers.forEach(t => clearTimeout(t))
})
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

.scrollbar-thin::-webkit-scrollbar { width: 4px; }
.scrollbar-thin::-webkit-scrollbar-track { background: rgba(30,41,59,0.5); }
.scrollbar-thin::-webkit-scrollbar-thumb { background: rgba(71,85,105,0.5); border-radius: 2px; }
.scrollbar-thin::-webkit-scrollbar-thumb:hover { background: rgba(100,116,139,0.5); }
</style>

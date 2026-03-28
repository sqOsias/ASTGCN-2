<template>
  <div class="h-screen flex flex-col bg-cyber-dark overflow-hidden">
    <!-- Header -->
    <header class="h-14 flex items-center justify-between px-6 border-b border-cyan-500/20 bg-cyber-darker/80">
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center">
            <el-icon :size="20"><Connection /></el-icon>
          </div>
          <h1 class="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            实时交通态势感知系统
          </h1>
        </div>
        <div class="h-6 w-px bg-cyan-500/30"></div>
        <span class="text-xs text-slate-400">ASTGCN Deep Learning Engine</span>
      </div>
      
      <div class="flex items-center gap-6">
        <!-- Connection Status -->
        <div class="flex items-center gap-2">
          <span :class="['status-dot', wsConnected ? 'status-green' : 'status-red']"></span>
          <span class="text-xs text-slate-400">{{ wsConnected ? 'WebSocket 已连接' : '连接中...' }}</span>
        </div>
        
        <!-- Virtual Time -->
        <div class="flex items-center gap-2 text-cyan-400">
          <el-icon><Clock /></el-icon>
          <span class="font-mono text-sm">{{ currentTime }}</span>
        </div>
        
        <!-- Simulation Controls -->
        <div class="flex items-center gap-2">
          <el-button 
            :type="isRunning ? 'danger' : 'success'" 
            size="small" 
            @click="toggleSimulation"
            circle
          >
            <el-icon><VideoPlay v-if="!isRunning" /><VideoPause v-else /></el-icon>
          </el-button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 overflow-hidden">
      <el-tabs v-model="activeTab" type="border-card" class="h-full flex flex-col">
        <el-tab-pane label="宏观路网拓扑" name="topology" class="h-full">
          <template #label>
            <span class="flex items-center gap-2">
              <el-icon><Share /></el-icon>
              <span>宏观路网拓扑</span>
            </span>
          </template>
          <TopologyView 
            :networkData="networkData" 
            :topology="topology"
            :metrics="systemMetrics"
            @node-click="handleNodeClick"
          />
        </el-tab-pane>
        
        <el-tab-pane label="微观时空推演" name="timeseries" class="h-full">
          <template #label>
            <span class="flex items-center gap-2">
              <el-icon><TrendCharts /></el-icon>
              <span>微观时空推演</span>
            </span>
          </template>
          <TimeSeriesView 
            :selectedNode="selectedNode"
            :networkData="networkData"
            :historyBuffer="historyBuffer"
            @select-node="handleNodeSelect"
          />
        </el-tab-pane>
        
        <el-tab-pane label="模型可解释性" name="attention" class="h-full">
          <template #label>
            <span class="flex items-center gap-2">
              <el-icon><Cpu /></el-icon>
              <span>模型可解释性</span>
            </span>
          </template>
          <AttentionView 
            :topology="topology"
            @node-pair-click="handleNodePairClick"
          />
        </el-tab-pane>
      </el-tabs>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, watch } from 'vue'
import TopologyView from './components/TopologyView.vue'
import TimeSeriesView from './components/TimeSeriesView.vue'
import AttentionView from './components/AttentionView.vue'

// State
const activeTab = ref('topology')
const wsConnected = ref(false)
const isRunning = ref(true)
const currentTime = ref('--:--:--')
const selectedNode = ref(0)

const systemMetrics = reactive({
  mae: 0,
  rmse: 0
})

const networkData = ref([])
const topology = reactive({
  nodes: [],
  edges: []
})

// History buffer for time series (stores last 48 frames)
const historyBuffer = ref([])
const MAX_HISTORY = 48

// WebSocket connection
let ws = null
let reconnectTimer = null

const connectWebSocket = () => {
  const wsUrl = `ws://${window.location.hostname}:8000/ws`
  console.log('Connecting to WebSocket:', wsUrl)
  
  ws = new WebSocket(wsUrl)
  
  ws.onopen = () => {
    console.log('WebSocket connected')
    wsConnected.value = true
    
    // Request topology data
    ws.send(JSON.stringify({ type: 'request_topology' }))
  }
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      
      if (data.type === 'topology') {
        topology.nodes = data.data.nodes
        topology.edges = data.data.edges
        return
      }
      
      if (data.type === 'pong') return
      
      // Stream data
      if (data.timestamp) {
        currentTime.value = data.timestamp
        systemMetrics.mae = data.system_metrics.current_mae
        systemMetrics.rmse = data.system_metrics.current_rmse
        networkData.value = data.network_status
        
        // Add to history buffer
        historyBuffer.value.push({
          timestamp: data.timestamp,
          data: data.network_status
        })
        
        // Trim history
        if (historyBuffer.value.length > MAX_HISTORY) {
          historyBuffer.value.shift()
        }
      }
    } catch (e) {
      console.error('Error parsing WebSocket message:', e)
    }
  }
  
  ws.onclose = () => {
    console.log('WebSocket disconnected')
    wsConnected.value = false
    
    // Reconnect after 3 seconds
    reconnectTimer = setTimeout(() => {
      connectWebSocket()
    }, 3000)
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
}

// Fetch initial topology
const fetchTopology = async () => {
  try {
    const res = await fetch('/api/topology')
    const data = await res.json()
    topology.nodes = data.nodes
    topology.edges = data.edges
  } catch (e) {
    console.error('Error fetching topology:', e)
  }
}

// Toggle simulation
const toggleSimulation = async () => {
  try {
    const endpoint = isRunning.value ? '/api/simulation/pause' : '/api/simulation/resume'
    await fetch(endpoint, { method: 'POST' })
    isRunning.value = !isRunning.value
  } catch (e) {
    console.error('Error toggling simulation:', e)
  }
}

// Handle node click from topology
const handleNodeClick = (nodeId) => {
  selectedNode.value = nodeId
  activeTab.value = 'timeseries'
}

// Handle node selection from time series view
const handleNodeSelect = (nodeId) => {
  selectedNode.value = nodeId
}

// Handle node pair click from attention view
const handleNodePairClick = (sourceId, targetId) => {
  console.log('Node pair clicked:', sourceId, targetId)
}

// Lifecycle
onMounted(() => {
  fetchTopology()
  connectWebSocket()
  
  // Ping to keep connection alive
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }))
    }
  }, 30000)
})

onUnmounted(() => {
  if (ws) ws.close()
  if (reconnectTimer) clearTimeout(reconnectTimer)
})
</script>

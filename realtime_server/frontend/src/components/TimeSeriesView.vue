<template>
  <Splitpanes class="h-full">
    <Pane :size="78" :min-size="30">
    <!-- Left: Charts Area -->
    <div class="h-full flex flex-col p-3 min-w-0 overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between mb-2 flex-shrink-0">
        <div class="flex items-center gap-3">
          <span class="text-slate-400 text-sm">选择节点:</span>
          <el-select 
            v-model="currentNode" 
            filterable 
            placeholder="选择节点"
            style="width: 130px"
            size="small"
            @change="handleNodeChange"
          >
            <el-option 
              v-for="i in 307" 
              :key="i - 1" 
              :label="`Node_${i - 1}`" 
              :value="i - 1"
            />
          </el-select>
        </div>
        <div class="flex items-center gap-4 text-xs">
          <div class="flex items-center gap-1.5">
            <span class="w-4 h-0.5 bg-cyan-400 rounded"></span>
            <span class="text-slate-400">真实车速</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="w-4 h-0.5 bg-orange-400 rounded" style="border-bottom: 2px dashed #fb923c;"></span>
            <span class="text-slate-400">预测车速</span>
          </div>
        </div>
      </div>

      <!-- Main chart + bottom charts: vertical split -->
      <Splitpanes horizontal class="flex-1 min-h-0">
        <Pane :size="70" :min-size="20">
          <!-- Main Chart -->
          <div class="h-full cyber-panel">
            <div class="cyber-panel-header py-1.5 text-sm">
              <el-icon><TrendCharts /></el-icon>
              Node_{{ currentNode }} 真实 vs 预测车速
            </div>
            <v-chart 
              ref="mainChartRef"
              class="w-full"
              style="height: calc(100% - 32px);"
              :option="mainChartOption" 
              :autoresize="true"
            />
          </div>
        </Pane>
        <Pane :size="30" :min-size="10">
          <!-- Bottom Charts Row: horizontal split -->
          <Splitpanes class="h-full">
            <Pane :size="50" :min-size="15">
              <div class="h-full cyber-panel">
                <div class="cyber-panel-header py-1 text-xs">
                  <el-icon><Histogram /></el-icon>
                  预测误差
                </div>
                <v-chart 
                  ref="residualChartRef"
                  class="w-full"
                  style="height: calc(100% - 24px);"
                  :option="residualChartOption" 
                  :autoresize="true"
                />
              </div>
            </Pane>
            <Pane :size="50" :min-size="15">
              <div class="h-full cyber-panel">
                <div class="cyber-panel-header py-1 text-xs">
                  <el-icon><Aim /></el-icon>
                  多步预测对比
                </div>
                <v-chart 
                  class="w-full"
                  style="height: calc(100% - 24px);"
                  :option="multiStepChartOption" 
                  :autoresize="true"
                />
              </div>
            </Pane>
          </Splitpanes>
        </Pane>
      </Splitpanes>
    </div>
    </Pane>

    <Pane :size="22" :min-size="8">
    <!-- Right Panel -->
    <Splitpanes horizontal class="h-full">
      <Pane :size="30" :min-size="10">
        <!-- Current Status -->
        <div class="h-full p-2 overflow-auto">
          <div class="cyber-panel h-full">
            <div class="cyber-panel-header py-1.5 text-sm">
              <el-icon><InfoFilled /></el-icon>
              节点状态
            </div>
            <div class="p-2">
              <div class="text-center mb-2">
                <div class="text-3xl font-bold" :class="getSpeedClass(currentSpeed)">
                  {{ currentSpeed.toFixed(0) }}
                </div>
                <div class="text-slate-400 text-xs">km/h</div>
              </div>
              <div class="grid grid-cols-2 gap-1.5 text-xs">
                <div class="bg-slate-800/50 rounded p-1.5 text-center">
                  <div class="text-cyan-400">{{ historyAvg.toFixed(1) }}</div>
                  <div class="text-slate-500">均值</div>
                </div>
                <div class="bg-slate-800/50 rounded p-1.5 text-center">
                  <div class="text-cyan-400">{{ mae.toFixed(2) }}</div>
                  <div class="text-slate-500">MAE</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Pane>
      <Pane :size="50" :min-size="15">
        <!-- Future Predictions -->
        <div class="h-full p-2 overflow-auto">
          <div class="cyber-panel h-full flex flex-col">
            <div class="cyber-panel-header py-1.5 text-sm flex-shrink-0">
              <el-icon><Timer /></el-icon>
              未来1h预测
            </div>
            <div class="p-2 space-y-1 overflow-y-auto flex-1">
              <div 
                v-for="(speed, index) in futurePredictions" 
                :key="index"
                class="flex items-center gap-2 text-xs"
              >
                <span class="text-slate-500 w-12">+{{ (index + 1) * 5 }}m</span>
                <div class="flex-1 h-1.5 bg-slate-800 rounded overflow-hidden">
                  <div 
                    class="h-full transition-all duration-300"
                    :class="getSpeedBgClass(speed)"
                    :style="{ width: Math.min(100, speed / 1.8) + '%' }"
                  ></div>
                </div>
                <span class="w-10 text-right font-mono" :class="getSpeedClass(speed)">
                  {{ speed.toFixed(0) }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </Pane>
      <Pane :size="20" :min-size="8">
        <!-- Trend -->
        <div class="h-full p-2 overflow-auto">
          <div class="cyber-panel h-full">
            <div class="cyber-panel-header py-1 text-xs">
              <el-icon><DataLine /></el-icon>
              趋势
            </div>
            <div class="p-2 text-center">
              <span class="font-bold text-lg" :class="trendClass">{{ trendLabel }}</span>
              <div class="text-xs text-slate-500">
                预计{{ trendDelta > 0 ? '+' : '' }}{{ trendDelta.toFixed(1) }} km/h
              </div>
            </div>
          </div>
        </div>
      </Pane>
    </Splitpanes>
    </Pane>
  </Splitpanes>
</template>

<script setup>
import { ref, computed, watch, shallowRef } from 'vue'
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart } from 'echarts/charts'
import { 
  TooltipComponent, 
  GridComponent, 
  MarkLineComponent,
  LegendComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import { TrendCharts, Histogram, InfoFilled, Timer, DataLine, Aim } from '@element-plus/icons-vue'

use([
  CanvasRenderer, 
  LineChart, 
  BarChart, 
  TooltipComponent, 
  GridComponent, 
  MarkLineComponent,
  LegendComponent
])

const props = defineProps({
  selectedNode: { type: Number, default: 0 },
  networkData: { type: Array, default: () => [] },
  historyBuffer: { type: Array, default: () => [] }
})

const emit = defineEmits(['select-node'])

const mainChartRef = shallowRef(null)
const residualChartRef = shallowRef(null)
const currentNode = ref(props.selectedNode)

watch(() => props.selectedNode, (newVal) => {
  currentNode.value = newVal
})

const handleNodeChange = (val) => {
  emit('select-node', val)
}

const getSpeedClass = (speed) => {
  if (speed > 50) return 'text-green-400'
  if (speed > 20) return 'text-yellow-400'
  return 'text-red-400'
}

const getSpeedBgClass = (speed) => {
  if (speed > 50) return 'bg-green-500'
  if (speed > 20) return 'bg-yellow-500'
  return 'bg-red-500'
}

const currentNodeData = computed(() => {
  return props.networkData.find(n => n.node_id === currentNode.value) || null
})

const currentSpeed = computed(() => {
  return currentNodeData.value?.current_real_speed || 0
})

const futurePredictions = computed(() => {
  return currentNodeData.value?.future_pred_speeds || []
})

// History data with both real and predicted values
const historyData = computed(() => {
  return props.historyBuffer.map((frame, index) => {
    const nodeData = frame.data?.find(n => n.node_id === currentNode.value)
    return {
      index,
      timestamp: frame.timestamp,
      real: nodeData?.current_real_speed || 0,
      predicted: nodeData?.future_pred_speeds?.[0] || 0
    }
  })
})

// Calculate MAE
const mae = computed(() => {
  const validData = historyData.value.filter(d => d.real > 0 && d.predicted > 0)
  if (validData.length === 0) return 0
  const sum = validData.reduce((acc, d) => acc + Math.abs(d.real - d.predicted), 0)
  return sum / validData.length
})

const historyAvg = computed(() => {
  const speeds = historyData.value.map(d => d.real).filter(s => s > 0)
  return speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : 0
})

// Trend calculation
const trendDelta = computed(() => {
  if (futurePredictions.value.length < 2) return 0
  const lastPred = futurePredictions.value[futurePredictions.value.length - 1]
  return lastPred - currentSpeed.value
})

const trendClass = computed(() => {
  if (trendDelta.value > 5) return 'text-green-400'
  if (trendDelta.value < -5) return 'text-red-400'
  return 'text-yellow-400'
})

const trendLabel = computed(() => {
  if (trendDelta.value > 5) return '↑ 向好'
  if (trendDelta.value < -5) return '↓ 恶化'
  return '→ 平稳'
})

// Main chart option - Show real vs predicted comparison
const mainChartOption = computed(() => {
  const histLen = historyData.value.length
  const futureLen = futurePredictions.value.length
  
  // X-axis labels
  const xLabels = []
  for (let i = 0; i < histLen; i++) {
    xLabels.push(`-${(histLen - i) * 5}m`)
  }
  xLabels.push('NOW')
  for (let i = 1; i <= futureLen; i++) {
    xLabels.push(`+${i * 5}m`)
  }
  
  // Real speed: history + current + null for future
  const realData = historyData.value.map(d => d.real)
  realData.push(currentSpeed.value)
  for (let i = 0; i < futureLen; i++) {
    realData.push(null)
  }
  
  // Predicted speed: history predictions + current prediction + future predictions
  // This shows the predicted curve overlaid on the real curve
  const predData = historyData.value.map(d => d.predicted)
  predData.push(futurePredictions.value[0] || currentSpeed.value)
  futurePredictions.value.forEach(p => predData.push(p))
  
  // Fixed Y-axis range 0-180
  const yMin = 0
  const yMax = 180
  
  return {
    backgroundColor: 'transparent',
    grid: {
      left: 60,
      right: 20,
      top: 25,
      bottom: 35,
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter: (params) => {
        let result = `<div style="font-weight:bold;margin-bottom:4px">${params[0].axisValue}</div>`
        params.forEach(p => {
          if (p.value !== null && p.value !== undefined) {
            result += `<div>${p.marker} ${p.seriesName}: <b>${p.value.toFixed(1)}</b> km/h</div>`
          }
        })
        return result
      }
    },
    xAxis: {
      type: 'category',
      data: xLabels,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { 
        color: '#64748b', 
        fontSize: 10,
        interval: (index) => index % 6 === 0 || xLabels[index] === 'NOW'
      },
      splitLine: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'km/h',
      nameTextStyle: { color: '#64748b', fontSize: 10 },
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#64748b', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1e293b' } },
      min: yMin,
      max: yMax
    },
    series: [
      {
        name: '真实车速',
        type: 'line',
        data: realData,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#22d3ee', width: 2.5 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(34, 211, 238, 0.2)' },
              { offset: 1, color: 'rgba(34, 211, 238, 0)' }
            ]
          }
        }
      },
      {
        name: '预测车速',
        type: 'line',
        data: predData,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: '#f97316', width: 2, type: 'dashed' }
      }
    ]
  }
})

// Residual chart option
const residualChartOption = computed(() => {
  const residuals = historyData.value.map(d => {
    const err = d.real - d.predicted
    return {
      value: err,
      itemStyle: {
        color: err > 0 ? '#22d3ee' : '#f97316'
      }
    }
  })
  
  // Fixed range for residuals
  const range = 50
  
  return {
    backgroundColor: 'transparent',
    grid: {
      left: 55,
      right: 15,
      top: 15,
      bottom: 20,
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: historyData.value.map((_, i) => i),
      show: false
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: '#64748b', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1e293b' } },
      min: -range,
      max: range,
      interval: 25
    },
    series: [{
      type: 'bar',
      data: residuals,
      barWidth: '50%',
      markLine: {
        silent: true,
        symbol: 'none',
        data: [
          { yAxis: 0, lineStyle: { color: '#475569', width: 1 } }
        ]
      }
    }]
  }
})

// Multi-step prediction comparison chart
const multiStepChartOption = computed(() => {
  const predictions = futurePredictions.value
  const steps = ['5m', '15m', '30m', '45m', '60m']
  const indices = [0, 2, 5, 8, 11] // Corresponding indices in future_pred_speeds array
  
  const data = indices.map((idx, i) => ({
    name: steps[i],
    value: predictions[idx] || 0
  }))
  
  return {
    backgroundColor: 'transparent',
    grid: {
      left: 40,
      right: 15,
      top: 10,
      bottom: 20,
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#e2e8f0', fontSize: 11 },
      formatter: (params) => `${params[0].name}: <b>${params[0].value?.toFixed(1)}</b> km/h`
    },
    xAxis: {
      type: 'category',
      data: steps,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#64748b', fontSize: 9 }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: '#64748b', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1e293b' } },
      min: 0,
      max: 140
    },
    series: [{
      type: 'line',
      data: data.map(d => d.value),
      smooth: true,
      symbol: 'circle',
      symbolSize: 6,
      lineStyle: { color: '#f97316', width: 2 },
      itemStyle: { color: '#f97316', borderColor: '#0a0e1a', borderWidth: 1 },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(249, 115, 22, 0.3)' },
            { offset: 1, color: 'rgba(249, 115, 22, 0)' }
          ]
        }
      },
      markLine: {
        silent: true,
        symbol: 'none',
        data: [
          { yAxis: currentSpeed.value, lineStyle: { color: '#22d3ee', width: 1, type: 'dashed' } }
        ],
        label: { show: false }
      }
    }]
  }
})
</script>

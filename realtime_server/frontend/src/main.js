import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'
import './style.css'

console.log('[main.js] Creating Vue app...')

const app = createApp(App)

app.config.errorHandler = (err, vm, info) => {
  console.error('[Vue Error]', err, info)
  document.getElementById('app').innerHTML = `<pre style="color:red;padding:20px;">${err}\n${err.stack}\n\nInfo: ${info}</pre>`
}

// Register all Element Plus icons
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(ElementPlus, { size: 'default', zIndex: 3000 })

try {
  app.mount('#app')
  console.log('[main.js] Vue app mounted successfully')
} catch (e) {
  console.error('[main.js] Mount failed:', e)
  document.getElementById('app').innerHTML = `<pre style="color:red;padding:20px;">Mount failed: ${e}\n${e.stack}</pre>`
}

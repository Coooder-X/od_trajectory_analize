import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import store from './store/store'

createApp(App).use(ElementPlus).use(store).mount('#app')

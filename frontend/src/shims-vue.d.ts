/* eslint-disable */
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// 下载 @type/mapbox-gl报错（部分包开发者可能没有上传自己的.d.ts代码到npm分支，这时会报错说找不到这个包，在shims-vue.d.ts添加即可
declare module 'mapbox-gl'
declare module 'd3'
declare module '@/types/vuex'
<template>
  <div class="map-container">
    <div id="map" ref="mapDivElement" class="map-content"></div>
    <cluster-layer v-if="initSuccess" :map="map" class="cluster-layer"></cluster-layer>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { defineComponent } from "vue";
import { onMounted, Ref, ref } from "vue";
import MapboxLanguage from "@mapbox/mapbox-gl-language"; // 加中文
import "mapbox-gl/dist/mapbox-gl.css";
import mapboxgl from 'mapbox-gl';
import { Marker, Map } from 'mapbox-gl/index'
import ClusterLayer from "./ClusterLayer.vue";

export default defineComponent({
  components: {
    ClusterLayer
  },
  name: 'MapComp',
  setup() {
    const mapDivElement: Ref<HTMLDivElement | null> = ref(null)
    const map: Ref<Map> = ref({}) as Ref<Map>
    const marker: Ref<Marker> = ref({}) as Ref<Marker>
    const initSuccess: Ref<Boolean> = ref(false)


    function mapNew(map: Ref<Map>, mapDivElement: Ref<HTMLDivElement | null>, marker: Ref<Marker>, arr: [number, number]) {
      if (mapDivElement.value !== null) {
        // console.log(mapDivElement.value)
        map.value = new mapboxgl.Map({
            container: mapDivElement.value, // container id 绑定的组件的id
            center: arr, // 初始坐标系
            minZoom: 1.7, // 设置最小拉伸比例
            zoom: 10, // starting zoom 地图初始的拉伸比例
            style: "mapbox://styles/mapbox/light-v9", // 类型light-v9
            // pitch: 60, //地图的角度，不写默认是0，取值是0-60度，一般在3D中使用
            bearing: 0, //地图的初始方向，值是北的逆时针度数，默认是0，即是正北,-17.6
            antialias: false, //抗锯齿，通过false关闭提升性能
            // maxBounds: [[, ], // 西南方坐标
            // [, ]] //东北方坐标,用于控制缩放范围
            attributionControl: false //  不展示 mapbox 的官网信息
        });

        map.value.addControl(new mapboxgl.NavigationControl(), "top-right");  //  导航

        map.value.addControl( //  中文包
          new MapboxLanguage({
            defaultLanguage: "zh-Hans",
          })
        );

        const scale = new mapboxgl.ScaleControl({
          maxWidth: 100,
          unit: 'metric'
        });
        map.value.addControl(scale, "bottom-left");// 比例尺
      }
    };
    const initMap = () => {
      mapboxgl.accessToken =
        "pk.eyJ1IjoidnVlamF2YSIsImEiOiJja3E3Zmc3cnAwNWl5Mm9yenZ4dmxrdnFlIn0.xskeHvMcXwPwOeg-3Unsjg";
      //pk.eyJ1Ijoiam9yZGl0b3N0IiwiYSI6ImQtcVkyclEifQ.vwKrOGZoZSj3N-9MB6FF_A
      mapNew(map, mapDivElement, marker, [120.094491, 30.239897]);
      initSuccess.value = true
    }
    
    onMounted(initMap);


    return {
      map, 
      mapDivElement,
      marker,
      initSuccess,
    };
  },
});
</script>

<style scoped>
.map-container {
  height: 100%;
  width: 100%;
  /* padding: 20px; */
}
.map-content {
  height: 100%;
  width: 100%;
}
.mapboxgl-ctrl.mapboxgl-ctrl-scale {
	height: 10px;
	background-color:transparent;
	line-height:10%;
	text-align:center
}
:deep(.mapboxgl-ctrl-logo) {
  display:none !important;
}

</style>

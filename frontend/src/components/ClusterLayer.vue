<template>
  <div>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { computed, defineComponent, PropType, watch } from "vue";
import { Ref, ref } from "vue";
import { useStore } from 'vuex';
import "mapbox-gl/dist/mapbox-gl.css";
import mapboxgl from 'mapbox-gl';
import { Map } from 'mapbox-gl/index'
import * as d3 from 'd3';

export default defineComponent({
  components: {},
  name: 'ClusterLayer',
  props: {
    map: {
      type: Object as PropType<Map>,  //  props 中引入自定义类型，需要用 PropType<>
      default: {}
    }
  },
  setup(props) {
    const clusterLayerSvg: Ref<any | null> = ref(null);
    const store = useStore();
    const { getters } = store;

    // let pointsExist = computed(() => getters.pointsExist);
    let odPoints = computed(() => getters.odPoints);
    // console.log(pointsExist, odPoints.value)

    //  监听 od 点数据变化，如果时间范围改变，则重新绘制 od 点
    watch(odPoints, (newValue: Array<[]>, oldValue: Array<[]>) => {
      if(!clusterLayerSvg.value) {
        initLayer();
      }
      paintLayer(clusterLayerSvg.value, odPoints.value);
    });

    const project = (d: Array<number>) => {
      return props.map.project(new mapboxgl.LngLat(d[0], d[1]));
    }
    
    //  初始化 od 点图层 svg
    const initLayer = () => {
      const container = props.map.getCanvasContainer();
      const svg = d3
        .select(container)
        .append("svg")
        .attr("width", "100%")
        .attr("height", "2000")
        .style("position", "absolute")
        .style("z-index", 2);

      //  将轨迹点图层的 svg 更新到 store
      clusterLayerSvg.value = svg
      store.commit('setClusterLayerSvg', svg);
    }

    const paintLayer = (svg: any, pointsData: Array<[]>) => {
      //  如果已存在绘制的 od 点，清空再绘制新的
      svg
        .selectAll("#od_points")
        .remove()

      // Add svg objects
      const dots = svg
        .selectAll("circle")
        .data(pointsData)
        .enter()
        .append("circle")
        .attr('id', 'od_points')
        .attr("r", 3)
        .attr('stroke', 'black')
        .attr('stroke-width', '1px')
        .style("opcaity", 0.7)
        .style("fill", "#ff3636");

      // Render method redraws circles
      function render() {
        dots
          .attr("cx", function(d: Array<number>) {
            return project(d).x;
          })
          .attr("cy", function(d: Array<number>) {
            return project(d).y;
          });
      }

      // Call render method, and whenever map changes
      render();
      props.map.on("viewreset", render);
      props.map.on("move", render);
      props.map.on("moveend", render);
    }


    return {

    };
  },
});
</script>

<style scoped>


</style>

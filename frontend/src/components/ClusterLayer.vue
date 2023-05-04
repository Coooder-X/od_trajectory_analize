<template>
  <div>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { computed, ComputedRef, defineComponent, onMounted, PropType, watch } from "vue";
import { Ref, ref } from "vue";
import { useStore } from 'vuex';
import "mapbox-gl/dist/mapbox-gl.css";
import mapboxgl from 'mapbox-gl';
import { Map, PointLike } from 'mapbox-gl/index'
import { MapMode } from '@/map-interface'
import * as d3 from 'd3';
import {colorTable} from '@/color-pool'
import { useBrush, debounce, useDrawODPath } from "@/hooks/gisLayerHooks";

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

    let partOdPoints: ComputedRef<number[][]> = computed(() => getters.partOdPoints);
    let pointClusterMap = computed(() => getters.pointClusterMap);
    let clusterPointMap = computed(() => getters.clusterPointMap);
    let cidCenterMap = computed(() => getters.cidCenterMap);
    let odIndexList = computed(() => getters.odIndexList);
    const mapMode = computed(() => getters.mapMode);

    const project = (d: [number, number]) => {
      return props.map.project(new mapboxgl.LngLat(d[0], d[1]));
    }

    const { drawODPath, moveOdPath, setMarker, updateArrow } = useDrawODPath(project, clusterLayerSvg);

    const unproject = (d: [number, number]) => {
      return props.map.unproject({x: d[0], y: d[1]} as PointLike);
    }

    const { 
      setBrushLayerVisible,
      selectedSvgs,
      noSelectedSvgs,
      selectedODIdxs,
      selectedClusterIdxs
    } = useBrush({clusterLayerSvg, project, unproject});

    watch(mapMode, () => {
      setBrushLayerVisible(mapMode.value.has(MapMode.SELECT));
    }, { deep: true }); //  watch 监听 Set 对象内容必须添加 deep: true，否则只会监听 Set 对象本身的变化，而不是它的元素的变化

    watch([selectedODIdxs, selectedClusterIdxs],
      debounce(() => {  //  节流，不然每次 brush 调用都存 store，性能非常差
        store.commit('setSelectedODIdxs', selectedODIdxs.value);
        store.commit('setSelectedClusterIdxs', selectedClusterIdxs.value);
        const adjTable: {[key: number]: number[]} = {};
        getters.outAdjTable.forEach(function(value: number[], key: number) {
          adjTable[key] = value;
        });
        //  获取每个簇中心的坐标
        const clusterPointObj: {[key: number]: number[]} = {}
        clusterPointMap.value.forEach(function(value: number[], key: number) {
          clusterPointObj[key] = value;
        });
        store.dispatch('getLineGraph', {
          selectedClusterIdxs: selectedClusterIdxs.value,
          outAdjTable: adjTable,
          cluster_point_dict: clusterPointObj,
        });
        // store.dispatch('getCidCenterMap', {
        //   cluster_point_dict: clusterPointObj,
        //   selected_cluster_idxs: selectedClusterIdxs.value,
        // });
      }, 1000)
    );

    watch([mapMode], () => {
      if(cidCenterMap.value && clusterLayerSvg.value) {
        drawODPath(cidCenterMap.value); //  hover力导向节点后，簇之间的 path
        updateArrow();
      }
    }, {deep: true});

    //  监听 od 点数据变化，如果时间范围改变，则重新绘制 od 点
    watch(partOdPoints, (newValue: number[][], oldValue: number[][]) => {
      if(!clusterLayerSvg.value) {
        initLayer();
      }
      paintLayer(clusterLayerSvg.value, partOdPoints.value);
    });

    watch(pointClusterMap, () => {
      paintLayer(clusterLayerSvg.value, partOdPoints.value);
    })

    //  初始化 od 点图层 svg
    const initLayer = () => {
      d3.selectAll('svg').remove();
      clusterLayerSvg.value = null;
      const container = props.map.getCanvasContainer();
      const svg = d3
        .select(container)
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .style("position", "absolute")
        .style("z-index", 2);
      
      let g = svg.append('g').attr('id', 'point_group')
      const svgPath = svg.append('g').attr('id', 'paths');
      setMarker(svgPath); //  只在 initLayer 时给 paths 图层设置一次箭头的配置（多次设置则箭头消失），便于后续在link上添加箭头

      //  将轨迹点图层的 svg 更新到 store
      clusterLayerSvg.value = svg
      store.commit('setClusterLayerSvg', svg);
    }

    const paintLayer = (svg: any, pointsData: number[][]) => {
      //  如果已存在绘制的 od 点，清空再绘制新的
      svg
        .selectAll("circle")
        .remove()

      let g = svg.select('#point_group')

      // Add svg objects
      let dots = g
        .selectAll("circle")
        .data(pointsData)
        .enter()
        .append("circle")
        .attr('id', function(d: any, i: number) {
          return odIndexList.value[i];
        })
        .attr("r", 4)
        .attr('stroke', 'black')
        .attr('stroke-width', '0.5px')
        .style("opcaity", 0.7)
        .style("fill", function(point: number[], i: number) {
          const index = odIndexList.value[i]
          if(pointClusterMap.value.has(index))
            return colorTable[pointClusterMap.value.get(index)];
          return "#ff3636"
        });

      drawODPath(cidCenterMap.value); //  hover力导向节点后，簇之间的 path

      // Render method redraws circles
      function render() {
        dots
          .attr("cx", function(d: [number, number]) {
            return project(d).x;
          })
          .attr("cy", function(d: [number, number]) {
            return project(d).y;
          });

        moveOdPath(); //  和上面 dots 一样，拖动地图时需要更新 path 的位置，同步移动，同时绑定箭头。
        updateArrow();
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

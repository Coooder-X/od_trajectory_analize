<template>
  <div>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { defineComponent, PropType } from "vue";
import { onMounted, Ref, ref } from "vue";
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
    // const mapDivElement: Ref<HTMLDivElement | null> = ref(null)

    const project = (d: Array<number>) => {
      return props.map.project(new mapboxgl.LngLat(d[0], d[1]));
    }

    onMounted(() => {
      initLayer()
    });
    
    const initLayer = () => {
      var container = props.map.getCanvasContainer();
      var svg = d3
        .select(container)
        .append("svg")
        .attr("width", "100%")
        .attr("height", "2000")
        .style("position", "absolute")
        .style("z-index", 2);

      // Add data
      var data = [[120.094491, 30.239897], [120.194491, 30.339897], [120.064491, 30.29897]];

      // Add svg objects
      var dots = svg
        .selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("r", 5)
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

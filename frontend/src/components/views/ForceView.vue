<template>
  <div class="force-view">
    <view-header viewId="C" title="力导向视图"></view-header>
    <div id="force" ref="forceDivElement" class="force-content"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed, onMounted, watch, ComputedRef } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import * as d3 from "d3";
import ViewHeader from "../ViewHeader.vue";
import { ForceLink, ForceNode } from "@/global-interface";

export default defineComponent({
  components: {
    ViewHeader,
  },
  name: "ForceView",
  props: {},
  setup() {
    const width = 700;
    const height = 675;
    const store = useStore();
    const { getters } = store;
    const forceSvg: Ref<any | null> = ref(null);
    const groupSvg: Ref<any | null> = ref(null);
    const forceTreeLinks: ComputedRef<ForceLink> = computed(
      () => getters.forceTreeLinks
    );
    const forceTreeNodes: ComputedRef<ForceNode> = computed(
      () => getters.forceTreeNodes
    );

    onMounted(() => {
      initLayer();
    });

    watch([forceTreeLinks, forceTreeNodes], () => {
      if(forceTreeNodes && forceTreeLinks) {
        initLayer();
        drawGraph(forceTreeLinks.value, forceTreeNodes.value);
      }
    });

    const initLayer = () => {
      const container = document.getElementById("force");
      if(forceSvg.value) {
        forceSvg.value = d3
          .select(container)
          .selectAll("svg")
          .remove(); 
        forceSvg.value = null;
      }
      forceSvg.value = d3
        .select(container)
        .append("svg")
        .attr("width", `${width}px`)
        .attr("height", `${height}px`)
        .style("position", "absolute")
        .style("z-index", 2);

      groupSvg.value = forceSvg.value
        .append("g")
        .attr("transform", "translate(" + 0 + "," + 0 + ")");
    };

    const drawGraph = (edges: any, nodes: any) => {
      groupSvg.value
        .append("g")
        .selectAll("line").remove()
      groupSvg.value
        .append("g")
        .selectAll("circle").remove()
      groupSvg.value
        .selectAll(".circleText").remove()
      const colorScale = d3
        .scaleOrdinal()
        .domain(d3.range(nodes.length))
        .range(d3.schemeCategory10);

      const forceSimulation = d3
        .forceSimulation()
        .force(
          "link",
          d3.forceLink().id(function (d: any) {
            return d.name;
          })
        )
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter());

      //生成节点数据
      const points = forceSimulation.nodes(nodes).on("tick", ticked); //这个函数很重要，后面给出具体实现和说明

      //生成边数据
      forceSimulation
        .force("link")
        .links(edges)
        .distance(function (d: any) {
          //每一边的长度
          // if(!d.isFake)
          return 30; //d.value*100;
          // return null
        });
      // .strength(function(d){
      // 	// if(d.isFake)
      // 	// 	return 0
      // 	return 100
      // })

      //设置图形的中心位置
      forceSimulation
        .force("center")
        .x(width / 2)
        .y(height / 2);

      //绘制边
      const links = groupSvg.value
        .append("g")
        .selectAll("line")
        .data(edges)
        .enter()
        .append("line")
        .attr("stroke", function (d: any, i: number) {
          // if(d.isFake)
          return colorScale(i);
          // return null// 'transparent'
        })
        .attr("stroke-width", 1);

      const gs = groupSvg.value
        .selectAll(".circleText")
        .data(nodes)
        .enter()
        .append("g")
        .attr("transform", function (d: any, i: number) {
          const cirX = d.x;
          const cirY = d.y;
          return "translate(" + cirX + "," + cirY + ")";
        })
        .call(
          d3.drag().on("start", started).on("drag", dragged).on("end", ended)
        );

      //绘制节点
      gs.append("circle")
        .attr("r", 4)
        .attr("fill", function (d: any, i: number) {
          return colorScale(i);
        });
      //文字
      // gs.append("text")
      //   .attr("x", -10)
      //   .attr("y", -20)
      //   .attr("dy", 10)
      //   .text(function (d: any) {
      //     return d.name;
      //   });

      function ticked() {
        links
          .attr("x1", function (d: any) {
            return d.source.x;
          })
          .attr("y1", function (d: any) {
            return d.source.y;
          })
          .attr("x2", function (d: any) {
            return d.target.x;
          })
          .attr("y2", function (d: any) {
            return d.target.y;
          });

        // linksText
        // 	.attr("x",function(d){
        // 	return (d.source.x+d.target.x)/2;
        // })
        // .attr("y",function(d){
        // 	return (d.source.y+d.target.y)/2;
        // });

        gs.attr("transform", function (d: any) {
          return "translate(" + d.x + "," + d.y + ")";
        });
      }

      function started(d: any) {
        if (!d3.event) {
          forceSimulation.alphaTarget(0.8).restart(); //设置衰减系数，对节点位置移动过程的模拟，数值越高移动越快，数值范围[0，1]
        }
        d.subject.fx = d.subject.x;
        d.subject.fy = d.subject.y;
      }
      function dragged(d: any) {
        d.subject.fx = d.x;
        d.subject.fy = d.y;
      }
      function ended(d: any) {
        if (!d3.event) {
          forceSimulation.alphaTarget(0);
        }
        d.subject.fx = null;
        d.subject.fy = null;
      }
    };

    return {};
  },
});
</script>

<style scoped>
.force-view {
  height: calc(500px + 200px + 10px);
  width: 700px;
  margin-bottom: 10px;
  background-color: white;
}

.force-content {
  width: 100%;
  height: 100%;
}
</style>

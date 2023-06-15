<template>
    <div class="force-view">
      <view-header viewId="C" title="力导向视图-数据集可视化"></view-header>
      <div id="force" ref="forceDivElement" class="force-content">
        <div class="toolbar">
          <el-button @click="onClick">可视化数据集
          </el-button>
        </div>
      </div>
    </div>
  </template>
  
<script lang="ts">
/* eslint-disable */
import { defineComponent, computed, onMounted, watch, ComputedRef, nextTick } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import * as d3 from "d3";
import { PointLike } from 'mapbox-gl/index'
import ViewHeader from "../ViewHeader.vue";
import { ForceLink, ForceNode } from "@/global-interface";
import { useGetCircleByCluster } from "@/hooks/gisLayerHooks";
import { getHullPaths, updateGroups } from "@/hooks/polygonHullHooks";
import PolarHeatMap from "../PolarHeatMap.vue";
import PoiPanel from "../PoiPanel.vue";
import { MapMode } from "@/map-interface";
import { calLenColor, calNodeColor } from "@/utils";
import axios from "axios";
  
const StrokeWidth = 0.2;
const color = ['red', 'green', 'blue']
  
  export default defineComponent({
    components: {
      ViewHeader,
      PolarHeatMap,
      PoiPanel,
    },
    name: "ForceView4vis",
    props: {},
    setup() {
      const width = 700;
      const height = 675;
      const store = useStore();
      const { getters } = store;
      const forceSvg: Ref<any | null> = ref(null);
      const groupSvg: Ref<any | null> = ref(null);
      const labels: Ref<number[]> = ref([]);
      let forceSimulation: any;
      const forceTreeLinks: ComputedRef<ForceLink> = computed(
        () => getters.forceTreeLinks
      );
      const forceTreeNodes: ComputedRef<ForceNode> = computed(
        () => getters.forceTreeNodes
      );
      const odIndexList = computed(() => getters.odIndexList);
      const odPoints = computed(() => getters.odPoints);
      const cidCenterMap = computed(() => getters.cidCenterMap);
      const clusterPointMap = computed(() => getters.clusterPointMap);
      const partClusterPointMap = computed(() => getters.partClusterPointMap);
      const { getCircleByClusterId } = useGetCircleByCluster();
      const svgCircles: Ref<any> = ref(null);
      const clusterLayerSvg = computed(() => getters.clusterLayerSvg);
      const forceNodeSvg: Ref<any | null> = ref(null);
      const map = computed(() => getters.map);
      const communityGroup = computed(() => getters.communityGroup);
      const colorTable = computed(() => getters.colorTable);
      const dateScope = computed(() => getters.dateScope);
  
      watch([forceTreeLinks, forceTreeNodes], () => {
        if(forceTreeNodes && forceTreeLinks) {
          initLayer();
          drawGraph(forceTreeLinks.value, forceTreeNodes.value);
        }
      });
  
      const initLayer = () => {
        if(forceSimulation) {
          forceSimulation.stop();
          forceSimulation = null;
        }
        const container = document.getElementById("force");
        if(forceSvg.value) {
          forceSvg.value.selectAll('#hull_paths').remove();
          forceSvg.value = d3.select(container).selectAll("svg").remove();
          forceSvg.value = null;
        }
        forceSvg.value = d3
          .select(container)
          .append("svg")
          .attr("width", `${width}px`)
          .attr("height", `${height}px`)
          .style("position", "absolute")
          .style("z-index", 2)
        //   .on('click', onMouseClick('close'));
  
        groupSvg.value = forceSvg.value
          .append("g")
          .attr("transform", "translate(" + 0 + "," + 0 + ")");
  
        groupSvg.value.selectAll('#hull_paths').remove();
  
        //  画布的拖动和缩放
        //创建一个缩放行为
        const zoom = d3
          .zoom()
          .scaleExtent([0.5, 10]) //设置缩放范围
          .on("zoom", zoomed); //设置缩放事件
        //为 SVG 添加缩放行为
        forceSvg.value.call(zoom);
        //缩放事件处理函数
        function zoomed(event: any) {
          groupSvg.value.attr("transform", event.transform); //更新 g 元素的 transform 属性
        }
      };
  
      const drawGraph = (edges: any, nodes: any) => {
        const [startDay, endDay] = [dateScope.value[0]+1, dateScope.value[1]+1];
  
        forceSimulation = d3
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
  
        //设置图形的中心位置
        forceSimulation
          .force("center")
          .x(width / 2)
          .y(height / 2);
  
        //  获取凸包，社区检测的结果
        let paths: any = null;
        if (communityGroup.value) {
          paths = getHullPaths(communityGroup.value, groupSvg.value);
        }
  
        //绘制边
        const links = groupSvg.value
          .append("g")
          .selectAll("line")
          .data(edges)
          .enter()
          .append("line")
          .attr("stroke", function (d: any, i: number) {
            return 'gray'
          })
          .style('display', function(d: any) {  //  fake 边不显示，不触发事件
            if (d.isFake)
              return 'none';
            else
              return 'visible';
          })
          .attr("stroke-width", StrokeWidth)
  
        const gs = groupSvg.value
          .append('g')
          .attr('id', 'force-nodes')
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
  
        //绘制节点,TODO: 颜色改为数据获取
        svgCircles.value = gs.append("circle")
          .attr("r", function(d: any) {
            return d.avgLen || 4;
          })
          .attr("fill", function (d: any, i: number) {
            return color[labels.value[d.name]]
          })
  
        function ticked() {
          links.attr("d", function(d: any) {
            const dx = d.target.x - d.source.x,//增量
                dy = d.target.y - d.source.y,
                dr = Math.pow(Math.sqrt(dx * dx + dy * dy), 1.1);
            return "M" + d.source.x + "," 
            + d.source.y + "A" + dr + "," 
            + dr + " 0 0,1 " + d.target.x + "," 
            + d.target.y;
          });
  
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
  
          // 绘制凸包，距离结果
          updateGroups(communityGroup.value, paths, svgCircles.value);
  
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

      const onClick = () => {
        console.log('alsdjfalksdjf')
        axios.get('/api/getGccDataVis', {}).then((res) => {
          console.log('res', res)
          const data = res.data;
          labels.value = data.G;
          initLayer();
          drawGraph(data.force_edges, data.force_nodes);
        });
      }
  
      return {
        forceNodeSvg,
        onClick,
      };
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
    height: calc(100% - 35px);
  }
  
  .toolbar {
    overflow: hidden;
    border: 3px silver solid;
    border-radius: 5px;
    margin-left: 3px;
    margin-top: 3px;
    width: 130px;
    padding: 5px;
    display: inline-block;
    /* display: inline-flex; */
    /* vertical-align: middle; */
    /* float: left; */
    position: absolute;
    background-color: white;
    z-index: 5000;
  }
  </style>
  
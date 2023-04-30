<template>
  <div class="force-view">
    <view-header viewId="C" title="力导向视图"></view-header>
    <div id="force" ref="forceDivElement" class="force-content">
    </div>
    <polar-heat-map
      v-if="headMapVisible"
      :svg="forceNodeSvg">
    </polar-heat-map>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed, onMounted, watch, ComputedRef, nextTick } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import * as d3 from "d3";
import ViewHeader from "../ViewHeader.vue";
import { ForceLink, ForceNode } from "@/global-interface";
import { useGetCircleByCluster } from "@/hooks/gisLayerHooks";
import PolarHeatMap from "../PolarHeatMap.vue";
import { MapMode } from "@/map-interface";

export default defineComponent({
  components: {
    ViewHeader,
    PolarHeatMap,
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
    let forceSimulation: any;
    const forceTreeLinks: ComputedRef<ForceLink> = computed(
      () => getters.forceTreeLinks
    );
    const forceTreeNodes: ComputedRef<ForceNode> = computed(
      () => getters.forceTreeNodes
    );
    const odIndexList = computed(() => getters.odIndexList);
    const pointClusterMap = computed(() => getters.pointClusterMap);
    const { getCircleByClusterId } = useGetCircleByCluster();
    const svgCircles: Ref<any> = ref(null);
    const clusterLayerSvg = computed(() => getters.clusterLayerSvg);
    const headMapVisible: Ref<Boolean> = ref(false);
    const forceNodeSvg: Ref<any | null> = ref(null);

    onMounted(() => {
      // initLayer();
    });

    watch([forceTreeLinks, forceTreeNodes], () => {
      if(forceTreeNodes && forceTreeLinks) {
        initLayer();
        drawGraph(forceTreeLinks.value, forceTreeNodes.value);
      }
    });

    const toggleShowMapOdPair = (srcCid: number, tgtCid: number, isAdd: boolean) => {
      //  hover 时让地图视图出现OD对，用 sessionStorate 传递参数
      store.commit('toggleMapMode', MapMode.CHOOSE_POINT);
      if (isAdd) {
        sessionStorage.setItem('odPair', JSON.stringify({
          srcCid, tgtCid
        }));
      } else {
        sessionStorage.removeItem('odPair');
      }
    }

    const onMouseOver = function(event: 'over' | 'out') {
      let isOver = (event === 'over');
      let radius = isOver? 6 : 4;
      return function(_: MouseEvent, d: any) {
        const self = d3.select(this);
        self.attr('r', radius).style('cursor', isOver? 'pointer':'default');
        const {name} = d;
        const [sourceCid, targetCid] = name.split('_').map(Number);
        toggleShowMapOdPair(sourceCid, targetCid, isOver);
        const odCircles = clusterLayerSvg.value.selectAll("circle")
        const c1 = getCircleByClusterId(odCircles, sourceCid);
        const c2 = getCircleByClusterId(odCircles, targetCid);
        console.log('from-to =', name);
        c1.attr('r', radius);
        c2.attr('r', radius);
      }
    }

    const onMouseClick = function(event: 'open' | 'close') {
      let isOpen = (event === 'open');
      let radius = isOpen? 6 : 4;
      return function(_: MouseEvent, d: any) {
        const self = d3.select(this);
        const odCircles = clusterLayerSvg.value.selectAll("circle")
        if (isOpen) {
          //  处理地图点的高亮
          const {name} = d;
          const [sourceCid, targetCid] = name.split('_').map(Number);
          const c1 = getCircleByClusterId(odCircles, sourceCid);
          const c2 = getCircleByClusterId(odCircles, targetCid);
          console.log('from-to =', name);
          c1.attr('r', radius);
          c2.attr('r', radius);
          //  处理极坐标热力图的显示
          // const regex = /translate\((\d+\.\d+),(\d+\.\d+)\)/;
          // const [__, x, y] = d3.select(self.node().parentNode).attr('transform').match(regex);
          forceNodeSvg.value = d3.select(self.node().parentNode);
        } else {
          odCircles.attr('r', 4);
          forceNodeSvg.value = null;
        }
        _.stopPropagation();
        if (isOpen && headMapVisible.value) {
          headMapVisible.value = false;
        }
        headMapVisible.value = isOpen;
      }
    }

    const initLayer = () => {
      if(forceSimulation) {
        forceSimulation.stop();
        forceSimulation = null;
      }
      const container = document.getElementById("force");
      if(forceSvg.value) {
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
        .on('click', onMouseClick('close'));

      groupSvg.value = forceSvg.value
        .append("g")
        .attr("transform", "translate(" + 0 + "," + 0 + ")");

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
      // groupSvg.value.append("g").selectAll("line").remove();
      // groupSvg.value.append("g").selectAll("circle").remove();
      // groupSvg.value.selectAll(".circleText").remove();
      const colorScale = d3
        .scaleOrdinal()
        .domain(d3.range(nodes.length))
        .range(d3.schemeCategory10);

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
        .distance(function (d: any) {
          //每一边的长度
          // if(!d.isFake)
          return 30; //d.value*100;
          // return null
        })
        .strength(function (d: any) {
          // if(d.isFake)
          // 	return 0
          return 0.9;
        });

      //设置图形的中心位置
      forceSimulation
        .force("center")
        .x(width / 2)
        .y(height / 2);

      //定义一个箭头
      groupSvg.value
        .append("svg:defs")
        .append("marker")
        .attr("id", "arrow") //设置箭头的 id，用于引用
        .attr("viewBox", "-0 -5 10 10")
        .attr("refX", 17) //设置箭头距离节点的距离
        .attr("refY", 0) //设置箭头在 y 轴上的偏移量
        .attr("orient", "auto") //设置箭头随边的方向旋转
        .attr("markerWidth", 4) //设置箭头的宽度
        .attr("markerHeight", 4) //设置箭头的高度
        .attr("xoverflow", "visible")
        .append("svg:path")
        .attr("d", "M 0,-5 L 10 ,0 L 0,5") //使用绝对坐标来绘制三角形
        .attr("fill", function (d: any, i: number) {
          return colorScale(i);
        })
        .style("stroke", function (d: any, i: number) {
          return colorScale(i);
        });

      //绘制边
      const links = groupSvg.value
        .append("g")
        .selectAll("line")
        .data(edges)
        .enter()
        .append("line")
        // .selectAll("path")
        // .data(edges)
        // .enter()
        // .append("path")
        .attr("marker-end", "url(#arrow)")
        .attr("stroke", function (d: any, i: number) {
          // if(d.isFake)
          return colorScale(i);
          // return null// 'transparent'
        })
        .attr("stroke-width", 1)
        // .attr("fill", "none");

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
        // .call(
        //   d3.drag().on("start", started).on("drag", dragged).on("end", ended)
        // );

      //绘制节点,TODO: 颜色改为数据获取
      svgCircles.value = gs.append("circle")
        .attr("r", 4)
        .attr("fill", function (d: any, i: number) {
          return colorScale(i);
        })
        .on('mouseover', onMouseOver('over'))
        .on('mouseout', onMouseOver('out'))
        .on('click', onMouseClick('open'))

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
          // links.attr("d", function(d: any) {
          //   var dx = d.target.x - d.source.x,//增量
          //       dy = d.target.y - d.source.y,
          //       dr = Math.sqrt(dx * dx + dy * dy);
          //   return "M" + d.source.x + "," 
          //   + d.source.y + "A" + dr + "," 
          //   + dr + " 0 0,1 " + d.target.x + "," 
          //   + d.target.y;
          // });

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

    return {
      forceNodeSvg,
      headMapVisible,
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
  height: 100%;
}
</style>

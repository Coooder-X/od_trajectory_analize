<template>
  <div class="force-view">
    <view-header viewId="C" title="力导向视图"></view-header>
    <div id="force" ref="forceDivElement" class="force-content">
      <div class="toolbar">
        考虑地理空间关系：
        <el-switch :value="withSpaceDist" @change="changeWithSpaceDist">
        </el-switch>
      </div>
    </div>
    <polar-heat-map
      v-if="headMapVisible"
      :svg="forceNodeSvg">
    </polar-heat-map>
    <poi-panel v-if="poiPanelVisible" style="z-index: 5000;"></poi-panel>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { defineComponent, computed, onMounted, watch, ComputedRef, nextTick } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import * as d3 from "d3";
import ViewHeader from "../ViewHeader.vue";
import { ForceLink, ForceNode } from "@/global-interface";
import { useGetCircleByCluster } from "@/hooks/gisLayerHooks";
import { getHullPaths, updateGroups } from "@/hooks/polygonHullHooks";
import PolarHeatMap from "../PolarHeatMap.vue";
import PoiPanel from "../PoiPanel.vue";
import { MapMode } from "@/map-interface";
import { colorTable } from "@/color-pool";
import { calLenColor, calNodeColor } from "@/utils";

const StrokeWidth = 3;

export default defineComponent({
  components: {
    ViewHeader,
    PolarHeatMap,
    PoiPanel,
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
    const odPoints = computed(() => getters.odPoints);
    const cidCenterMap = computed(() => getters.cidCenterMap);
    const clusterPointMap = computed(() => getters.clusterPointMap);
    const { getCircleByClusterId } = useGetCircleByCluster();
    const svgCircles: Ref<any> = ref(null);
    const clusterLayerSvg = computed(() => getters.clusterLayerSvg);
    const headMapVisible: Ref<Boolean> = ref(false);
    const poiPanelVisible: Ref<Boolean> = ref(false);
    const forceNodeSvg: Ref<any | null> = ref(null);
    const map = computed(() => getters.map);
    const communityGroup = computed(() => getters.communityGroup);
    const withSpaceDist = computed(() => getters.withSpaceDist);  //  线图是否考虑空间距离

    watch([withSpaceDist, forceTreeLinks, forceTreeNodes], () => {
      if(forceTreeNodes && forceTreeLinks) {
        initLayer();
        drawGraph(forceTreeLinks.value, forceTreeNodes.value);
      }
    });

    const changeWithSpaceDist = (value: boolean) => {
      store.commit('setWithSpaceDist', value);
    }

    const toggleShowMapOdPair = (pairCids: [number, number][], isAdd: boolean) => {
      //  hover 时让地图视图出现OD对，用 sessionStorate 传递参数
      store.commit('toggleMapMode', MapMode.CHOOSE_POINT);
      if (isAdd) {
        sessionStorage.setItem('odPairs', JSON.stringify(
          pairCids
        ));
      } else {
        sessionStorage.removeItem('odPairs');
      }
    }

    const onMouseOverLink = (event: 'over' | 'out') => {
      let isOver = (event === 'over');
      let radius = isOver? 6 : 4;
      let strokeWidth = isOver? StrokeWidth * 1.5 : StrokeWidth;
      return function(_: MouseEvent, d: any) {
        _.stopPropagation();
        d3.select(this)
          .attr('stroke-width', strokeWidth)
          .style('cursor', isOver? 'pointer':'default');

        //  联动地图视图的簇高亮
        const {source} = d;
        const transCid: number = parseInt(source.name.split('_')[1]);
        const odCircles = clusterLayerSvg.value.selectAll("circle")
        const c = getCircleByClusterId(odCircles, transCid);
        c.attr('r', radius);

        //  将和 hover 到的边代表着相同簇的边也高亮加粗/恢复
        d3.selectAll('#force-links').filter((d: any) => {
          const { source: {name} } = d;
          const cid: number = parseInt(name.split('_')[1]);
          return cid === transCid;
        }).attr('stroke-width', strokeWidth)

        //  hover 一个 link 时，所有代表着相同簇的边所关联到的点（OD对）都在地图视图展示
        const cidPairs: [number, number][] = [];
        d3.selectAll('#force-links').filter((d: any) => {
          const { source: {name} } = d;
          const cid: number = parseInt(name.split('_')[1]);
          return cid === transCid;
        }).each((d: any) => {
          const { source: {name: name1}, target: {name: name2} } = d;
          const [src1, tgt1] = name1.split('_').map(Number);
          const [src2, tgt2] = name2.split('_').map(Number);
          cidPairs.push([src1, tgt1], [src2, tgt2]);
        });
        toggleShowMapOdPair(cidPairs, isOver);
      }
    }

    const onMouseOver = function(event: 'over' | 'out') {
      let isOver = (event === 'over');
      let mapRadius = isOver? 6 : 4;
      return function(_: MouseEvent, d: any) {
        const self = d3.select(this);
        const r = parseInt(d.avgLen);
        let radius = isOver? r * 1.5 : r;
        self.attr('r', radius).style('cursor', isOver? 'pointer':'default');
        const {name} = d;
        const [sourceCid, targetCid] = name.split('_').map(Number);
        toggleShowMapOdPair([[sourceCid, targetCid]], isOver);
        const odCircles = clusterLayerSvg.value.selectAll("circle")
        const c1 = getCircleByClusterId(odCircles, sourceCid);
        const c2 = getCircleByClusterId(odCircles, targetCid);
        console.log('from-to =', name, 'trjNum', d.trjNum, 'r', d.avgLen);
        c1.attr('r', mapRadius);
        c2.attr('r', mapRadius);
      }
    }

    const onMouseClick = function(event: 'open' | 'close') {
      let isOpen = (event === 'open');
      let radius = isOpen? 6 : 4;
      return function(_: MouseEvent, d: any) {
        _.stopPropagation();
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
          poiPanelVisible.value = false;
        }
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
        .on('click', onMouseClick('close'));

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
      const nodeColorMap = calNodeColor(nodes, clusterPointMap.value, odPoints.value);
      nodes = calLenColor(nodes, cidCenterMap.value, map.value);
      if (!withSpaceDist.value)
        edges = edges.filter((edge: any) => !edge.isFake);
      // else
      //   edges = edges.filter((edge: any) => !edge.singleFake);
      // const colorScale = d3
      //   .scaleOrdinal()
      //   .domain(d3.range(nodes.length))
      //   .range(d3.schemeCategory10);

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
          if (withSpaceDist.value)
            return d.value * 1.7;
          return 50;
        })
        .strength(function (d: any) {
          return 0.2;
        });

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

      //定义一个箭头
      groupSvg.value
        .append("svg:defs")
        .selectAll('marker')
        .data(edges)
        .enter()
        .append("marker")
        .attr("id", function(d: any, i: number) {
          return `arrow_${i}`
        }) //设置箭头的 id，用于引用
        // .attr("id", "arrow") //设置箭头的 id，用于引用
        .attr("viewBox", "-0 -5 10 10")
        .attr("refX", function(d: any) { //设置箭头距离节点的距离
          return 7 + d.target.avgLen;
        })
        .attr("refY", 0) //设置箭头在 y 轴上的偏移量
        .attr("orient", "auto") //设置箭头随边的方向旋转
        .attr("markerWidth", 4) //设置箭头的宽度
        .attr("markerHeight", 4) //设置箭头的高度
        .attr("xoverflow", "visible")
        .append("svg:path")
        .attr("d", "M 0,-5 L 10 ,0 L 0,5") //使用绝对坐标来绘制三角形
        .attr("fill", function (d: any, i: number) {
          const { source } = d;
          const { name } = source;
          const cid = parseInt(name.split('_')[1]);
          return colorTable[cid];
        })

      //绘制边
      const links = groupSvg.value
        .append("g")
        // .selectAll("line")
        // .data(edges)
        // .enter()
        // .append("line")
        .selectAll("path")
        .data(edges)
        .enter()
        .append("path")
        .attr('id', 'force-links')
        // .attr("marker-end", "url(#arrow)")
        .attr("marker-end", function(d: any, i: number) {
          return `url(#arrow_${i})`
        })
        .attr("stroke", function (d: any, i: number) {
          const { source } = d;
          const { name } = source;
          const cid = parseInt(name.split('_')[1]);
          return colorTable[cid];
        })
        .style('display', function(d: any) {  //  fake 边不显示，不触发事件
          if (d.isFake)
            return 'none';
          else
            return 'visible';
        })
        .attr("stroke-width", StrokeWidth)
        .on('mouseover', onMouseOverLink('over'))
        .on('mouseout', onMouseOverLink('out'))
        .on('click', function(_: MouseEvent, d: any) {
          _.stopPropagation();  //  防止 click 后又同时出发 forceSvg 的点击空白处 click
          poiPanelVisible.value = true;
        })
        .attr("fill", "none");

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
        .attr("r", function(d: any) {
          return d.avgLen || 2;
        })
        .attr("fill", function (d: any, i: number) {
          // return colorScale(i);
          return nodeColorMap.get(d.name);
        })
        .attr("stroke-width", 0.8)
        .attr('stroke', 'black')
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
      poiPanelVisible,
      withSpaceDist,
      changeWithSpaceDist,
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
  width: 190px;
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

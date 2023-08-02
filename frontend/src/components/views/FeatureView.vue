<template>
  <div class="feature-view">
    <view-header viewId="E" title="轨迹特征视图"></view-header>
    <div id="features" class="feature-view-content">

    </div>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, onMounted, Ref, ref, watch } from "vue";
import { useStore } from "vuex";
import * as echarts from "echarts";
import ecStat from "echarts-stat";
import * as d3 from 'd3';
import * as d3ScaleChromatic from 'd3-scale-chromatic';
import ViewHeader from "../ViewHeader.vue";

export default defineComponent({
  components: {
    ViewHeader,
  },
  name: "FeatureView",
  props: {},
  setup(props) {
    const store = useStore();
    const { getters } = store;
    const tsneResult = computed(() => getters.tsneResult);
    const featureLabels = computed(() => getters.featureLabels);
    const communityGroup = computed(() => getters.communityGroup);
    const groupIds = computed(() => [...communityGroup.value.keys()]);
    const relatedNodeNames = computed(() => getters.relatedNodeNames);
    const trjIdxs = computed(() => getters.trjIdxs);

    type EChartsOption = echarts.EChartsOption;
    let chartDom: any;
    let myChart: any;
    let option: EChartsOption;

    // 创建一个长度为 20 的输入域
    const domain = d3.range(20);
    // 创建一个输出范围，使用 d3.schemeCategory10 颜色方案
    const range = d3.schemeCategory10;

    // 创建一个序数比例尺，将输入域映射到输出范围
    const colorScale = d3.scaleOrdinal()
      .domain(domain)
      .range(range);

    // 生成一个长度为 20 的颜色数组
    const colorArray = domain.map((d: any) => colorScale(d));

    echarts.registerTransform((ecStat as any).transform.clustering);
    // echarts.registerTransform(transformXYZ as ExternalDataTransform);

    onMounted(() => {
      chartDom = document.getElementById("features")!;
      myChart = echarts.init(chartDom);
    });

    watch([tsneResult, featureLabels], () => {
      console.log('wath tsneResult')
      const CLUSTER_COUNT = [...new Set(featureLabels.value)].length;
      const DIENSIION_CLUSTER_INDEX = 2;
      const COLOR_ALL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]//colorArray; //['#ff0000', '#ffa500', '#ffff00', '#80ff00', '#00ff00', '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff', '#ff0080', '#808080', '#c0c0c0', '#ffffff', '#000000', '#800000', '#008000', '#000080', '#808000'];
      const mp = new Map();
      const pieces = [];
      for (let i = 0; i < tsneResult.value.length; ++i) {
        // let curColor = '';
        // let label = 0;
        // for (const [groupId, nodeNameList] of communityGroup.value) {
        //   if (nodeNameList.includes(relatedNodeNames.value[i])) {
        //     curColor = COLOR_ALL[groupId];
        //     label = groupId;
        //   }
        // }
        mp.set(COLOR_ALL[featureLabels.value[i]], (mp.get(COLOR_ALL[featureLabels.value[i]]) || 0) + 1)
        pieces.push({
          value: featureLabels.value[i],
          label: "community " + featureLabels.value[i],
          color: COLOR_ALL[featureLabels.value[i]],
        });
      }
      console.log('color map', mp)
      console.log('pieces', pieces)

      option = {
        dataset: [
          {
            source: tsneResult.value.map((item: any, i: number) => [...item, featureLabels.value[i]]),
          },
          {
            transform: {
              type: "ecStat:clustering",
              // print: true,
              config: {
                clusterCount: CLUSTER_COUNT,
                outputType: "single",
                outputClusterIndexDimension: DIENSIION_CLUSTER_INDEX,
              },
            },
          },
        ],
        series: [
          {
            type: "scatter",
            encode: {
              x: 0, // 第0个维度作为x坐标
              y: 1, // 第1个维度作为y坐标
              value: 2 // 第0个维度作为数据值
            },
            symbolSize: 15,
            itemStyle: {
              borderColor: "#555",
            },
          }
        ],
        tooltip: {
          position: "top",
          formatter: (d: any) => {
            console.log('i, ', d)
            return `TrjId: ${trjIdxs.value[d.dataIndex]}`
          }
        },
        visualMap: {
          type: "piecewise",
          top: "middle",
          min: 0,
          max: CLUSTER_COUNT,
          left: 10,
          splitNumber: CLUSTER_COUNT,
          dimension: DIENSIION_CLUSTER_INDEX,
          pieces: pieces,
        },
        grid: {
          left: 120,
          show: false,
        },
        xAxis: {
          show: false,
        },
        yAxis: {
          show: false,
        },
        // series: {
        //   type: "scatter",
        //   encode: { tooltip: [0, 1] },
        //   symbolSize: 15,
        //   itemStyle: {
        //     borderColor: "#555",
        //   },
        //   datasetIndex: 1,
        // },
      };

      option && myChart.setOption(option);
    });

    return {};
  },
});
</script>

<style scoped>
.feature-view {
  position: relative;
  width: 640px;
  height: 380px;
  top: calc(-200px - 500px - 10px - 380px);
  left: 920px;
  background-color: white;
}

.feature-view-content {
  height: 100%;
  padding-bottom: 30px;
  box-sizing: border-box;
}
</style>
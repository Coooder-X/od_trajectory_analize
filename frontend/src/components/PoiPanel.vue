<template>
  <div class="poi-panel">
    <div id="bar-chart" class="bar-chart"></div>
    <!-- <div class="dick-chart"></div>
    <div></div> -->
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import * as echarts from "echarts";
import * as d3 from "d3";
import axios from "axios";

export default defineComponent({
  components: {},
  name: "PoiPanel",
  props: {
    coords: {
      type: Array,
      required: true
    }
  },
  setup(props) {
    type EChartsOption = echarts.EChartsOption;
    let barChartDom: HTMLElement;
    let barChart: any;
    let barChartOption: EChartsOption;
    const colorMap: Map<string, string> = new Map();
    const barChartData: Ref<any> = ref([]);

    axios({
      method: 'post',
      url: '/api/getPoiInfoByPoint',
      data: {
        point_in_cluster: props.coords,
        radius: 160
      },
    }).then((res: any) => {
      console.log(res)
      barChartData.value = res.data['poi_type_dict'];
    });

    watch(barChartData, () => {
      barChartData.value.sort((a: any, b: any) => b.value - a.value)
      setChartOption(barChartData.value);
    });

    onMounted(() => {
      barChartDom = document.getElementById("bar-chart")!;
      barChart = echarts.init(barChartDom);

      // 创建一个序数比例尺，使用d3.schemeCategory10作为颜色域
      var color = d3
        .scaleOrdinal()
        .domain(barChartData.value.map((d: any) => d.type)) // 设置定义域为数据
        .range(d3.schemePaired.concat(["#808080", "#c0c0c0"])); // 设置值域为10种颜色

      // 输出每个数据对应的颜色
      barChartData.value.forEach(function (d: any, i: number) {
        colorMap.set(d.type, color(i));
      });
      setChartOption(barChartData.value);
    });

    const setChartOption = (data: any[]) => {
      barChartOption = {
        title: {
          text: "POI类型分布图",
        },
        xAxis: {
          type: "category",
          data: data.map((d: any) => d.type),
          axisTick: {
            show: true,
            
          },
          axisLabel: {
            show: true,
            interval: 0,
            rotate: -35
          }
        },
        yAxis: {
          type: "value",
          minInterval: 1
        },
        series: [
          {
            data: data.map((d: any) => {
              return {
                value: d.value,
                itemStyle: {
                  color: colorMap.get(d.type),
                },
              };
            }),
            type: "bar",
          },
        ],
        grid: {
          left: "38px",
          top: "30px",
          bottom: "45px",
        },
        legend: {
          type: "scroll",
          show: true,
        },
      };

      barChart.setOption(barChartOption);
      barChart.resize();
    };

    return {};
  },
});
</script>

<style scoped>
.poi-panel {
  width: 450px;
  /* height: 300px; */
  height: 180px;
  position: absolute;
  right: 10px;
  bottom: 10px;
  border: 1px solid black;
  border-radius: 7px;
  z-index: 5000;
  background-color: white;
}

.bar-chart {
  height: 160px;
  background-color: white;
  border-radius: 7px;
}

.dick-chart {
  height: 60px;
  background-color: rgb(218, 250, 215);
}
</style>
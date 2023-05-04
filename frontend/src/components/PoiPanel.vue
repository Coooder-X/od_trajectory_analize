<template>
  <div class="poi-panel">
    <div id="bar-chart" class="bar-chart"></div>
    <div class="dick-chart"></div>
    <div></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import * as echarts from "echarts";
import * as d3 from "d3";

export default defineComponent({
  components: {},
  name: "PoiPanel",
  props: {},
  setup(props) {
    type EChartsOption = echarts.EChartsOption;
    let barChartDom: HTMLElement;
    let barChart: any;
    let barChartOption: EChartsOption;
    const colorMap: Map<string, string> = new Map();

    onMounted(() => {
      barChartDom = document.getElementById("bar-chart")!;
      barChart = echarts.init(barChartDom);
      const data = [
        { type: "Mon", value: 12 },
        { type: "Tue", value: 34 },
        { type: "Wed", value: 54 },
        { type: "Thu", value: 10 },
        { type: "Fri", value: 32 },
        { type: "Sat", value: 35 },
        { type: "Sun", value: 122 },
        { type: "asd", value: 76 },
        { type: "sdf", value: 14 },
        { type: "agftr", value: 52 },
        { type: "asdw", value: 32 },
      ];
      const num = data.length;

      // 创建一个序数比例尺，使用d3.schemeCategory10作为颜色域
      var color = d3
        .scaleOrdinal()
        .domain(data.map(d => d.type)) // 设置定义域为数据
        .range(d3.schemePaired.concat(["#808080", "#c0c0c0"])); // 设置值域为10种颜色

      // 输出每个数据对应的颜色
      data.forEach(function (d: any, i: number) {
        colorMap.set(d.type, color(i));
      });
      setChartOption(data);
    });

    const setChartOption = (data: any[]) => {
      barChartOption = {
        title: {
          text: "POI类型分布图",
        },
        xAxis: {
          type: "category",
          data: data.map((d: any) => d.type),
        },
        yAxis: {
          type: "value",
        },
        series: [
          {
            data: data.map((d: any, i: number) => {
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
          bottom: "20px",
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
  height: 300px;
  position: absolute;
  right: 10px;
  bottom: 10px;
  border: 1px solid black;
  border-radius: 7px;
  z-index: 5000;
  background-color: white;
}

.bar-chart {
  height: 140px;
  background-color: white;
  border-radius: 7px;
}

.dick-chart {
  height: 60px;
  background-color: rgb(218, 250, 215);
}
</style>
<template>
  <div class="time-selector-container">
    <div ref="domRef" class="time-selector"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import * as echarts from "echarts";

export default defineComponent({
  components: {},
  name: "TimeSelector",
  props: {},
  setup() {
    type EChartsOption = echarts.EChartsOption;
    let chartDom; // = document.getElementById("time-selector")!;
    let timeSelector: any; // = echarts.init(chartDom);
    let option: EChartsOption;
    let data: Ref<any> = ref([]);
    const domRef: Ref<any> = ref(null);

    watch(data, () => {
      console.log("data", data);
    });

    onMounted(() => {
      chartDom = domRef.value;
      timeSelector = echarts.init(chartDom);
      let times = [];
      data.value = [];

      for (let i = 0; i < 24; i++) {
        times.push(i);
        data.value.push(Math.random());
      }

      option = {
        xAxis: {
          type: "category",
          data: times,
        },
        yAxis: {},
        dataZoom: [
          {
            filterMode: "filter",
            type: "inside",
            start: 0,
            end: 10,
          },
          {
            start: 0,
            end: 23,  //  改成 props 传
          },
        ],
        series: [
          {
            type: "line",
            data: data.value,
          },
        ],
      };

      timeSelector.setOption(option);

      //  获取选中的时间范围端点
      timeSelector.on("datazoom", function (params: any) {
        console.log(params); //里面存有代表滑动条的起始的数字
        let xAxis = timeSelector.getOption().dataZoom[0]; //获取axis
        console.log(xAxis.startValue); //滑动条左端对应在xAxis.data的索引
        console.log(xAxis.endValue); //滑动条右端对应在xAxis.data的索引
      });
    });
    
    return {
      domRef,
    }
  },
});
</script>

<style scoped>
.time-selector {
  bottom: 0px;
  left: -20px;  /*  改成 props 传 */
  position: absolute;
  width: 300px;
  height: 300px;
  overflow: hidden;
}
.time-selector-container {
  position: relative;
  width: 290px;
  height: 50px;
  /* background-color: antiquewhite; */
  overflow: hidden;
}
</style>
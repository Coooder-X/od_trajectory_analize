<template>
  <div class="time-selector-container">
    <div class="mask" v-if="disabled"></div>
    <div ref="domRef" class="time-selector" :class="{'time-selector-disabled': disabled}"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import * as echarts from "echarts";

export default defineComponent({
  components: {},
  name: "TimeSelector",
  props: {
    min: Number,
    max: Number,
    defaultMin: Number,
    defaultMax: Number,
    disabled: Boolean,
    dataRange: {
      type: Number,
      required: true,
    }
  },
  emits: ['change'],
  setup(props, { emit }) {
    const {dataRange} = props;
    type EChartsOption = echarts.EChartsOption;
    let chartDom: HTMLElement;
    let timeSelector: any; 
    let option: EChartsOption;
    let data: Ref<any> = ref([]);
    const domRef: Ref<any> = ref(null);

    let times: number[] = [];
    data.value = [];

    for (let i = dataRange === 31? 1:0; i <= (dataRange === 31? 31:23); i++) {
      times.push(i);
      data.value.push(Math.random());
    }

    onMounted(() => {
      chartDom = domRef.value;
      timeSelector = echarts.init(chartDom);

      //  获取选中的时间范围端点
      timeSelector.on("datazoom", 
        (params: any) => {
          console.log(params); //里面存有代表滑动条的起始的数字
          let xAxis = timeSelector.getOption().dataZoom[0]; //获取axis
          console.log(xAxis.startValue); //滑动条左端对应在xAxis.data的索引
          console.log(xAxis.endValue); //滑动条右端对应在xAxis.data的索引
          emit('change', [xAxis.startValue, xAxis.endValue]);
        }
      );
    });

    const setChartOption = () => {
      const { disabled, defaultMin, defaultMax, min, max } = props;
      option = {
        xAxis: {
          type: "category",
          data: times,
        },
        yAxis: {},
        dataZoom: [
          {
            filterMode: "filter",
            type: "slider",
            startValue: disabled? 0 : defaultMin,
            endValue: disabled? 0 : defaultMax,
            handleSize: 35,
            dataBackground: {
              areaStyle: {
                color: '#b4cdff'
              },
              lineStyle: {
                width: 2
              },
            },
            textStyle: {
              fontSize: 17
            }
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
    }

    watch(props, setChartOption, {deep: true});

    watch(domRef, setChartOption, {deep: true});
    
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
  width: 360px;
  height: 300px;
  overflow: hidden;
}
.time-selector-container {
  position: relative;
  width: 360px;
  height: 50px;
  left: -20px;
  /* background-color: antiquewhite; */
  overflow: hidden;
}

.mask {
  background-color: transparent;
  width: 100%;
  height: 100%;
  cursor: not-allowed;
  pointer-events: auto;
}

.time-selector-disabled {
  pointer-events: none;
}
</style>
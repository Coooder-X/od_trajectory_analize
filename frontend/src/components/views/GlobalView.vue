<template>
  <div class="global-view">
    <view-header viewId="A" title="地图视图"></view-header>
    <div class="global-view-content">
      <div class="file-comp">
        <b>数据集:</b>
        <el-select
          v-model="dataset"
          class="file-select"
          placeholder="选择数据集"
          @change="changeDataSet"
        >
          <el-option
            label="2020年5月份杭州轨迹数据集"
            :value="5"
          />
          <el-option
            label="2020年1月份杭州轨迹数据集"
            :value="1"
          />
        </el-select>
        <el-table
          cell-class-name="table_cell"
          :data="tableData"
          class="data-table"
          align="center"
          show-overflow-tooltip
          :show-header="false"
          :stripe="true"
          :border="true"
        >
          <el-table-column prop="name" label="FileName" width="250" />
        </el-table>
      </div>
      <el-divider direction="vertical" class="divider" />
      <div class="info-comp">
        <div class="selec-box">
          <div style="margin-bottom: 5px;">
            <b v-if="tableData.length">{{ `当前日期范围: ${dataset}月${dateScope[0]+1}日-${dataset}月${dateScope[1]+1}日` }}</b>
            <b v-else>选择日期范围:</b>
          </div>
          <time-selector
            :disabled="!tableData.length"
            :defaultMin="dateScope[0]"
            :defaultMax="dateScope[1]"
            :min="dateSelection[0]"
            :max="dateSelection.at(-1)"
            :dataRange="31"
            :data="dayTrjNums"
            @change="onChangeDateScope"></time-selector>
          <div style="margin-bottom: 5px;">
            <b v-if="tableData.length">{{ `当前时间范围: ${timeScope[0]}时-${timeScope[1]}时` }}</b>
            <b v-else>选择时间范围:</b>
            <time-selector
              :disabled="!tableData.length"
              :defaultMin="timeScope[0]"
              :defaultMax="timeScope[1]"
              :min="timeSelection[0]"
              :max="timeSelection.at(-1)"
              :data="hourTrjNums"
              :dataRange="24"
              @change="onChangeTimeScope"></time-selector>
          </div>
        </div>
        <div class="data-info">
          <b>全局信息：</b> <br>
          <b>{{ `${dateScope[0]+1}日-${dateScope[1]+1}日轨迹总数：` }}</b> <span></span>
          <b></b> <span></span>
          <b></b> <span></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
/* eslint-disable */
import { debounce } from "@/hooks/gisLayerHooks";
import axios from "axios";
import { defineComponent, computed, onMounted } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import TimeSelector from "../TimeSelector.vue";
import ViewHeader from "../ViewHeader.vue";

export default defineComponent({
  components: {
    ViewHeader,
    TimeSelector,
  },
  name: "GlobalView",
  props: {},
  setup() {
    const store = useStore();
    const dataset: Ref<number | null> = ref(null);
    let tableData: Ref<{ name: string }[]> = ref([]);
    const dateSelection: Ref<number[]> = ref([]);
    const timeSelection: Ref<number[]> = ref([]);
    const dateScope: Ref<[number, number]> = ref([] as any);
    const timeScope: Ref<[number, number]> = ref([] as any);
    const hourTrjNums: Ref<number[]> = ref([]);
    const dayTrjNums: Ref<number[]> = ref([]);
    let firstIn: Boolean = true;
    const {getters} = store;
    const month = computed(() => getters.month);

    store.dispatch('getAllODPoints', {params: {month: month.value, startDay: 1, endDay: 2}});

    const changeDataSet = () => {
      store.commit('setMonth', dataset.value);
      //  后面加上逻辑：修改数据集后，才显示 gis 轨迹点
      tableData.value = new Array(20).fill(0).map((_, index) => {
        return { name: `2020年${dataset.value}月${index + 1}日杭州市出租车GPS轨迹点数据.h5` };
      });
      dateSelection.value = new Array(20).fill(0).map((_, index) => index);
      timeSelection.value = new Array(24).fill(0).map((_, index) => index + 1);
      dateScope.value = [dateSelection.value[0], dateSelection.value[1]];
      timeScope.value = [timeSelection.value[7], timeSelection.value[9]];
      //  初始化时间后，第一次取数据初始化 gis 视图
      onTimeSelect(timeScope.value);
      onDateSelect(dateScope.value);
      getTrjNumByDay();
      getTrjNumByMonth();
    };

    const onTimeSelect = (event: [number, number]) => {
      console.log(event)
      timeScope.value[0] = event[0];
      timeScope.value[1] = event[1];
      store.commit('setTimeScope', timeScope.value);
      // store.dispatch('getODPointsFilterByHour', {params: {startHour: timeScope.value[0], endHour: timeScope.value[1]}});
      getODDataAction();
    }

    const onDateSelect = (event: [number, number]) => {
      console.log(event)
      dateScope.value[0] = event[0];
      dateScope.value[1] = event[1];
      store.commit('setDateScope', dateScope.value);
      if (firstIn) {
        firstIn = false;
      } else {
        getODDataAction();
      }
      getTrjNumByDay();
    }

    const onChangeTimeScope = debounce(onTimeSelect, 700);
    const onChangeDateScope = debounce(onDateSelect, 700);

    const getODDataAction = () => {
      store.dispatch('getODPointsFilterByDayAndHour', {
        params: {
          month: month.value,
          startDay: dateScope.value[0] + 1,
          endDay: dateScope.value[1] + 1,
          startHour: timeScope.value[0],
          endHour: timeScope.value[1],
        }
      });
    }

    const getTrjNumByDay = () => {
      axios.get('/api/getTrjNumByHour', {
        params: {
          month: dataset.value,
          startDay: dateScope.value[0] + 1,
          endDay: dateScope.value[1] + 1,
        }
      }).then((res: any) => {
        console.log('trj num day', res.data[dataset.value!]['nums'])
        hourTrjNums.value = res.data[dataset.value!]['nums'];
      });
    }

    const getTrjNumByMonth = () => {
      axios.get('/api/getTrjTotalNumByMonth', {
        params: {
          month: dataset.value,
        }
      }).then((res: any) => {
        console.log('trj num month', res.data)
        dayTrjNums.value = res.data;
      });
    }

    return {
      dataset,
      tableData,
      dateScope,
      timeScope,
      dateSelection,
      timeSelection,
      hourTrjNums,
      dayTrjNums,
      onChangeTimeScope,
      onChangeDateScope,
      changeDataSet,
      onTimeSelect,
    };
  },
});
</script>

<style scoped>
.global-view {
  --menu-width: 50px;
  height: 200px;
  width: 850px;
  background-color: white;
}

.global-view-content {
  height: calc(100% - var(--header-height));
  display: flex;
  justify-content: center;
}

.file-comp {
  height: 100%;
  width: 400px;
  padding: 10px;
}

.info-comp {
  height: 100%;
  width: 100%;
  display: flex;
  box-sizing: content-box;
}

.file-select {
  margin-left: 10px;
  width: 180px;
}

.data-table {
  height: 100px;
  width: 250px;
  margin-top: 10px;
}

:deep(.el-table) {
  border: 0.5px rgb(231, 231, 231) solid;
  border-radius: 5px;
}

:deep(.el-table .el-table__cell) {
  padding: 3px 0;
}

:deep(.table_cell .cell) {
  white-space: nowrap;
}

.divider {
  height: 90%;
  top: 5%;
  border-left: 3px #909399 solid;
}

.selec-box {
  margin: 10px 20px;
  width: 290px;
}

.data-info {
  margin: 5px;
  margin-left: -5px;
  margin-top: 7px;
  margin-right: 7px;
  border: 2px #bdc0c5 solid;
  border-radius: 5px;
  width: 100%;
  height: 90%;
}
</style>

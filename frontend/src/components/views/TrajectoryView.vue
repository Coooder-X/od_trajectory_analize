<template>
  <div class="trajectory-view">
    <view-header viewId="D" title="轨迹详情视图"></view-header>
    <div class="trajectory-view-content">
      <div class="trj-table">
        <el-table :data="trjDetails" stripe style="width: 100%; height: 100%;">
          <el-table-column prop="TrjId" label="TrjId" align="center"> </el-table-column>
          <el-table-column prop="startPoint" label="Start Point" width="175" align="center"> </el-table-column>
          <el-table-column prop="endPoint" label="End Point" width="175" align="center"> </el-table-column>
          <el-table-column prop="startTime" label="Start Time" align="center"> </el-table-column>
          <el-table-column prop="endTime" label="End Time" align="center"> </el-table-column>
          <el-table-column prop="avgSpeed" label="Avg Speed" align="center"> </el-table-column>
      </el-table>
      </div>
      <div class="trj-grid">
        <heat-grid v-for="(item, index) in trjSpeed" :key="item.TrjId" :TrjId="item.TrjId" :speedList="speedList[index]" ></heat-grid>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, onMounted, Ref, ref, watch } from "vue";
import ViewHeader from "../ViewHeader.vue";
import HeatGrid from "../HeatGrid.vue";
import { useStore } from "vuex";

export default defineComponent({
  components: {
    ViewHeader,
    HeatGrid,
  },
  name: "TrajectoryView",
  props: {
  },
  emits: ['change'],
  setup(props) {
    const store = useStore();
    const { getters } = store;
    const trjDetails = computed(() => getters.trjDetails.sort((a: any, b: any) => {a.TrjId - b.TrjId}));
    const trjSpeed = computed(() => getters.trjSpeed.sort((a: any, b: any) => {a.TrjId - b.TrjId}));
    const speedList = computed(() => trjSpeed.value.map((item: any) => item.speedList));
    console.log(trjSpeed, trjSpeed.value, speedList.value)
    
    return {
      trjDetails,
      trjSpeed,
      speedList,
    }
  },
});
</script>

<style scoped>
.trajectory-view {
  position: relative;
  width: 910px;
  height: 380px;
  top: calc(-200px - 500px - 10px);
  background-color: white;
}

.trajectory-view-content {
  height: 100%;
}

.trj-table {
  height: calc(50% - 5px - var(--header-height) / 2);
}

.trj-grid {
  height: calc(50% - 5px - var(--header-height) / 2);
  /* margin-top: 10px; */
  padding: 0px 10px 5px 10px;
  box-sizing: border-box;
  /* background-color: antiquewhite; */
  overflow: hidden;
}

.trj-grid:hover {
  overflow: auto;
  scrollbar-width: calc(5px);
}

.trj-grid::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}

.trj-grid::-webkit-scrollbar-thumb {
  background-color: silver;
  border-radius: 7px;
}

/* .trj-grid::-webkit-scrollbar {
  height: 20px;
} */
</style>
<template>
  <div class="trajectory-view">
    <view-header viewId="D" title="轨迹详情视图"></view-header>
    <div class="trajectory-view-content">
      <div class="trj-table">
        <el-table :data="tableData" stripe style="width: 100%; height: 100%;">
          <el-table-column prop="TrjId" label="TrjId"> </el-table-column>
          <el-table-column prop="startPoint" label="Start Point"> </el-table-column>
          <el-table-column prop="endPoint" label="End Point"> </el-table-column>
          <el-table-column prop="startTime" label="Start Time"> </el-table-column>
          <el-table-column prop="endTime" label="End Time"> </el-table-column>
          <el-table-column prop="avgSpeed" label="Avg Speed"> </el-table-column>
      </el-table>
      </div>
      <div class="trj-grid">
        <heat-grid></heat-grid>
        <heat-grid></heat-grid>
        <heat-grid></heat-grid>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, Ref, ref, watch } from "vue";
import ViewHeader from "../ViewHeader.vue";
import HeatGrid from "../HeatGrid.vue";

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
    
    
    return {
      
    }
  },
});
</script>

<style scoped>
.trajectory-view {
  position: relative;
  width: 700px;
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
  margin-top: 10px;
  padding: 0px 10px 5px 10px;
  box-sizing: border-box;
  background-color: antiquewhite;
}
</style>
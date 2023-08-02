<template>
  <div class="result-view">
    <view-header viewId="F" title="x视图"></view-header>
    <div class="result-view-content">
      
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
  name: "ResultyView",
  props: {
  },
  setup(props) {
    
    
    return {
      
    }
  },
});
</script>

<style scoped>
.result-view {
  position: relative;
  width: 200px;
  height: 380px;
  left: calc(710px + 650px);
  top: calc(-200px - 500px - 10px - 380px - 380px);
  background-color: white;
}

.result-view-content {
  height: 100%;
}
</style>
<template>
  <div class="force-view">
    <view-header viewId="C" title="力导向视图"></view-header>
    <div class="force-content">

    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed, onMounted } from "vue";
import { Ref, ref } from "vue";
import { useStore } from "vuex";
import ViewHeader from "../ViewHeader.vue";

export default defineComponent({
  components: {
    ViewHeader,
  },
  name: "ForceView",
  props: {},
  setup() {
    const store = useStore();
    

    return {
      
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
  height: 100%;
}

</style>

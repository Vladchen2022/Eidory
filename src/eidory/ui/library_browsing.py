"""Gallery browsing state and cancellable background database paging."""
import logging
import time
from weakref import ref
from PySide6.QtCore import QObject, Signal
from eidory.core.search_filters import SearchChainResult


class GalleryReader(QObject):
    loaded = Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self._window = ref(window)
        self.serial = 0
        self.loaded.connect(self.accept)

    def request(self, *, append, reveal_image_id=None):
        w = self._window()
        self.serial += 1
        serial = self.serial
        args = dict(text_query=w.current_keyword_query, status_filter=w._selected_status_filter(),
                    tag_ids=[], tag_match_mode='any', folder_path_prefix=w._selected_folder_path_prefix(),
                    collection_id=w._selected_collection_id(), virtual_filter=w._selected_virtual_filter(),
                    limit=w.page_size, offset=w.grid_view.rowCount() if append else 0,
                    sort_key=w._database_sort_key(), sort_desc=w.current_sort_desc)
        selected = w.grid_view.selected_image_ids()
        current = w.grid_view.current_image()
        current_id = current.id if current else None
        if not append:
            w.grid_view.set_images([])
        token = (serial, w.semantic_search_revision, w.grid_view.content_revision)
        w._loading_more_images = True
        w.statusBar().showMessage('正在载入图片…')
        def work():
            start = time.perf_counter()
            try:
                if reveal_image_id is not None:
                    position = w.store.image_position_in_collection(reveal_image_id, args['collection_id'],
                        sort_key=args['sort_key'], sort_desc=args['sort_desc'])
                    if position is not None:
                        args['limit'] = max(args['limit'], (position // w.page_size + 1) * w.page_size)
                result = w.store.list_images(**args)
            except Exception as exc:
                logging.getLogger('eidory.runtime').exception('Gallery query failed')
                result = exc
            elapsed = time.perf_counter() - start
            if elapsed > .08:
                logging.getLogger('eidory.runtime').info('Gallery query %.0fms offset=%s limit=%s', elapsed*1000, args['offset'], args['limit'])
            self.loaded.emit((token, append, reveal_image_id, args['limit'], result, selected, current_id))
        # Initial construction has no interactive window yet. Later reads never
        # block UI input. Tests can execute the same worker synchronously.
        if not w.isVisible():
            work()
        elif w._start_background_task(work, name='search-gallery') is False:
            w._loading_more_images = False

    def accept(self, payload):
        token, append, reveal, limit, result, selected, current_id = payload
        w = self._window()
        if w._closing or token[0] != self.serial:
            return
        w._loading_more_images = False
        if token[1] != w.semantic_search_revision or token[2] != w.grid_view.content_revision or w.current_result_mode != 'library':
            return
        if isinstance(result, Exception):
            w.statusBar().showMessage('图片载入失败，可切换文件夹后重试：' + str(result))
            return
        if append:
            w.grid_view.append_images(result)
        else:
            w.grid_view.set_images(result, selected_image_ids=selected, current_image_id=current_id)
        w.current_offset = w.grid_view.rowCount()
        w.load_more_button.setEnabled(len(result) >= limit)
        if reveal is not None:
            w.grid_view.select_image_id(reveal)
        w._set_result_status(f'已加载 {w.current_offset} 张')


class LibraryBrowsingMixin:
    def _reload_images(self, *, reveal_image_id: int | None = None) -> None:
        self.semantic_search_revision += 1
        if hasattr(self, "filter_workflow"):
            self.filter_workflow.follow_sidebar()
        self.search_button.setEnabled(True)
        self._invalidate_result_status_count_cache()
        self.current_offset = 0
        self._clear_manual_result_order()
        self._clear_result_management_state()
        self._clear_temporary_project_selection()
        self.search_filters.clear()
        self.active_filter_index = None
        self.current_keyword_query = None
        self.current_semantic_query = None
        self.current_result_mode = "library"
        self.current_semantic_images = []
        self.current_semantic_filtered_images = []
        self.current_similar_searchable_count = 0
        self.current_similar_candidate_limit = 0
        self.current_color_images = []
        self.current_color_filtered_images = []
        self.current_search_scope_count = None
        self.current_inspiration_project_id = None
        self.current_inspiration_terms = []
        self.current_inspiration_plan_filters = []
        self.current_inspiration_raw_term_results = []
        self.current_inspiration_images = []
        self.current_inspiration_filtered_images = []
        self.current_inspiration_matches = {}
        self.current_temp_project_id = None
        self.current_search_result_set_project_id = None
        self.current_temp_project_images = []
        self.current_temp_project_badges = {}
        self.current_creative_node_images = []
        self.current_creative_node_filtered_images = []
        self.current_creative_node_searchable_count = 0
        self.current_creative_node_candidate_limit = 0
        self.current_creative_node_badges = {}
        self.current_chain_images = []
        self.current_chain_filtered_images = []
        self.current_chain_result = SearchChainResult(images=[])
        self.current_chain_base_image_ids = None
        self.current_chain_base_label = None
        self.current_chain_operation_mode = "replace"
        self.current_duplicate_images = []
        self.current_duplicate_badges = {}
        self._show_gallery_view()
        self.load_more_button.setEnabled(True)
        self._gallery_request(append=False, reveal_image_id=reveal_image_id)
        self._update_search_diagnostics([])
        self._refresh_feedback_buttons(self.selected_image)
        self._refresh_filter_chain_ui()

    def _clear_temporary_project_selection(self) -> None:
        if not hasattr(self, "temp_project_list"):
            return
        self.temp_project_list.blockSignals(True)
        self.temp_project_list.clearSelection()
        self.temp_project_list.setCurrentItem(None)
        self.temp_project_list.blockSignals(False)

    def _load_more(self) -> None:
        if self._can_load_more_library_page():
            self._gallery_request(append=True)

    def _gallery_request(self, *, append, reveal_image_id=None):
        if not hasattr(self, '_gallery_reader'):
            self._gallery_reader = GalleryReader(self)
        self._gallery_reader.request(append=append, reveal_image_id=reveal_image_id)

    def _can_load_more_library_page(self) -> bool:
        return (
            not self._loading_more_images
            and self.current_result_mode == "library"
            and not self.search_filters
            and self.load_more_button.isEnabled()
        )

    def _maybe_auto_load_more(self, value: int) -> None:
        if not self._can_load_more_library_page():
            return
        bar = self.grid_view.verticalScrollBar()
        threshold = max(240, self.grid_view.viewport().height())
        if bar.maximum() - value <= threshold:
            self._load_more()

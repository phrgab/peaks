import pyqtgraph as pg


class _Crosshair(pg.GraphicsObject):
    """Crosshair class for pyqtgraph plots.
    Creates a crosshair with a draggable centre marker and spans for integration ranges.
    """

    def __init__(
        self,
        plot,
        pos=(0, 0),
        dim0_width=0,
        dim1_width=0,
        *,
        orientation="both",
        bounds=None,
        movable=True,
        symbol="crosshair",
        brush=None,
        axisOrder="col-major",
        target_item_kwargs=None,
        linear_region_item_kwargs=None,
    ):
        """
        Initializes a Crosshair instance for use in 2D pyqtgraph plots, providing a moveable crosshair with variable
         spans. Defiend by dim0 and dim1 properties.

        Parameters
        ----------
        plot : PlotItem
            The plot item to which this crosshair will be added.
        pos : tuple of float, optional
            Initial position (dim0, dim1) of the crosshair's center. Defaults to (0, 0).
        dim0_width : float
            Width of the crosshair lines in dim0 direction. Defaults to 0.
        dim1_width : float
            Width of the crosshair lines in dim1 direction. Defaults to 0.
        orientation : str, optional
            Orientation of the crosshair. Options are "horizontal", "vertical", or "both". Defaults to "both".
        bounds : list of tuple of float, optional
            Lower and upper bounds for the crosshair. Provide list of tuple
            [(min_bound_dim0, max_bound_dim0), (min_bound_dim1, max_bound_dim1)].
            Defaults to None.
        movable : bool, optional
            If True, allows the crosshair to be moved interactively. Defaults to True.
        symbol : str, optional
            Symbol used for the crosshair center marker. Defaults to "crosshair".
        brush : QBrush, optional
            Brush used for drawing the crosshair spans. Defaults to None.
        axisOrder : str, optional
            Order of the axes. Defaults to "col-major" for consistency with ImageItem default.
        target_item_kwargs : dict, optional
            Additional keyword arguments to pass to the TargetItem constructor. Defaults to None.
        linear_region_item_kwargs : dict, optional
            Additional keyword arguments to pass to the LinearRegionItem constructor. Defaults to None.

        Raises
        ------
        ValueError
            If an unsupported orientation or axisOrder is specified.
        """

        super().__init__()
        self.parent_plot = plot
        self.orientation = orientation
        self.axis_order = axisOrder
        self.dim0_width = dim0_width
        self.dim1_width = dim1_width
        self.lock_x = None
        self.lock_y = None
        if target_item_kwargs is None:
            target_item_kwargs = {}
        if linear_region_item_kwargs is None:
            linear_region_item_kwargs = {}

        # Define horizontal and vertical based on axisOrder
        if self.axis_order == "col-major":  # --> (h, v) = (dim0, dim1)
            self.h_width = self.dim0_width
            self.v_width = self.dim1_width
        elif self.axis_order == "row-major":  # --> (h, v) = (dim1, dim0)
            pos = (pos[1], pos[0])
            self.h_width = self.dim1_width
            self.v_width = self.dim0_width
        else:
            raise ValueError(
                "Invalid axisOrder. Only 'col-major' or 'row-major' are allowed."
            )

        if bounds:
            self.set_bounds(bounds)

        # Crosshair items
        self.xh = pg.TargetItem(
            pos=pos,
            movable=movable,
            symbol=symbol,
            **target_item_kwargs,
        )  # Marker
        # Spans
        if self.orientation == "both" or self.orientation == "horizontal":
            self.xh_h = pg.LinearRegionItem(
                orientation="horizontal",
                values=(
                    pos[1] - self.v_width / 2,
                    pos[1] + self.v_width / 2,
                ),
                brush=brush,
                movable=False,
                **linear_region_item_kwargs,
            )
        if self.orientation == "both" or self.orientation == "vertical":
            self.xh_v = pg.LinearRegionItem(
                orientation="vertical",
                values=(
                    pos[0] - self.h_width / 2,
                    pos[0] + self.h_width / 2,
                ),
                brush=brush,
                movable=False,
                **linear_region_item_kwargs,
            )
        else:
            raise ValueError(
                "Invalid orientation. Only 'horizontal', 'vertical' or 'both' are allowed."
            )

        # Adding items to the parent
        # Check if self.xh_v and self.xh_h exist
        if hasattr(self, "xh_v"):
            self.parent_plot.addItem(self.xh_v)
        if hasattr(self, "xh_h"):
            self.parent_plot.addItem(self.xh_h)
        self.parent_plot.addItem(self.xh)

        # Connect signals
        self.xh.sigPositionChanged.connect(self.update_crosshair)

    def update_crosshair(self):
        """Update the crosshair position and span based on the marker position."""
        x, y = self.xh.pos()  # New position in actual plot coordinates
        # Ensure in range if bounds are set
        if hasattr(self, "h_bounds"):
            x = min(
                self.h_bounds[1] - self.h_width / 2,
                max(self.h_bounds[0] + self.h_width / 2, x),
            )
        if hasattr(self, "v_bounds"):
            y = min(
                self.v_bounds[1] - self.v_width / 2,
                max(self.v_bounds[0] + self.v_width / 2, y),
            )

        # Lock the crosshair to a specific value if required
        if self.lock_x:
            x = self.lock_x
        if self.lock_y:
            y = self.lock_y

        # Set the crosshair position
        self.xh.setPos((x, y))

        # Set span of LinearRegion items
        if hasattr(self, "xh_h"):
            self.xh_h.setRegion([y - self.v_width / 2, y + self.v_width / 2])
        if hasattr(self, "xh_v"):
            self.xh_v.setRegion([x - self.h_width / 2, x + self.h_width / 2])

    def set_dim0_width(self, width):
        """Set the dim0 width of the crosshair lines.

        Parameters
        ----------
        width : float
            Width of the span of the crosshair lines for the dim0 axis.
        """
        x, y = self.xh.pos()  # Current position in plot coordinates
        if self.axis_order == "col-major":
            self.h_width = width
            if hasattr(self, "xh_h"):
                self.xh_v.setRegion([x - self.h_width / 2, x + self.h_width / 2])
        else:
            self.v_width = width
            if hasattr(self, "xh_v"):
                self.xh_h.setRegion([y - self.v_width / 2, y + self.v_width / 2])

    def set_dim1_width(self, width):
        """Set the dim1 width of the crosshair lines.

        Parameters
        ----------
        width : float
            Width of the span of the crosshair lines for the dim1 axis.
        """
        x, y = self.xh.pos()  # Current position in plot coordinates
        if self.axis_order == "col-major":
            self.v_width = width
            if hasattr(self, "xh_v"):
                self.xh_h.setRegion([y - self.v_width / 2, y + self.v_width / 2])
        else:
            self.h_width = width
            if hasattr(self, "xh_h"):
                self.xh_v.setRegion([x - self.h_width / 2, x + self.h_width / 2])

    def get_dim0_width(self):
        """Get the dim0 width of the crosshair lines.

        Returns
        -------
        float
            Width of the span of the crosshair lines for the dim0 axis.
        """
        return self.h_width if self.axis_order == "col-major" else self.v_width

    def get_dim1_width(self):
        """Get the dim1 span of the crosshair lines.

        Returns
        -------
        float
            Width of the span of the crosshair lines for the dim1 axis.
        """
        return self.v_width if self.axis_order == "col-major" else self.h_width

    def get_dim0_span(self):
        """Get the dim0 span of the crosshair lines.

        Returns
        -------
        tuple of float
            Span of the crosshair lines for the dim0 axis.
        """
        if self.axis_order == "col-major":
            return self.xh_v.getRegion()
        else:
            return self.xh_h.getRegion()

    def get_dim1_span(self):
        """Get the dim1 span of the crosshair lines.

        Returns
        -------
        tuple of float
            Span of the crosshair lines for the dim1 axis.
        """
        if self.axis_order == "col-major":
            return self.xh_h.getRegion()
        else:
            return self.xh_v.getRegion()

    def set_pos(self, pos):
        """Set the position of the crosshair center.

        Parameters
        ----------
        pos : tuple of float
            Position (x, y) of the crosshair's center.
        """
        if self.axis_order == "row-major":
            pos = (pos[1], pos[0])

        self.xh.setPos(pos)
        if hasattr(self, "xh_h"):
            self.xh_h.setRegion([pos[1] - self.v_width / 2, pos[1] + self.v_width / 2])
        if hasattr(self, "xh_v"):
            self.xh_v.setRegion([pos[0] - self.h_width / 2, pos[0] + self.h_width / 2])

    def get_pos(self):
        """Get the position of the crosshair center in (dim0, dim1) order.

        Returns
        -------
        tuple of float
            Position of the crosshair's center in (dim0, dim1) order."""
        if self.axis_order == "col-major":
            x, y = self.xh.pos()
        else:
            y, x = self.xh.pos()
        return x, y

    def set_bounds(self, bounds):
        """Set the bounds of the crosshair.

        Parameters
        ----------
        bounds : list of tuple of float
            Lower and upper bounds for the crosshair. Provide list of tuple
            [(min_bound_dim0, max_bound_dim0), (min_bound_dim1, max_bound_dim1)].
        """
        if self.axis_order == "col-major":
            self.h_bounds = sorted(bounds[0])
            self.v_bounds = sorted(bounds[1])
        else:
            self.h_bounds = sorted(bounds[1])
            self.v_bounds = sorted(bounds[0])

    def set_lock_dim0(self, value):
        """Lock the crosshair to a specific dim0 value.

        Parameters
        ----------
        value : float or None
            Value to lock the crosshair to. Set with None to unlock.
        """
        if self.axis_order == "col-major":
            self.lock_x = value
        else:
            self.lock_y = value

    def set_lock_dim1(self, value):
        """Lock the crosshair to a specific dim1 value. Set with None to unlock."""
        if self.axis_order == "col-major":
            self.lock_y = value
        else:
            self.lock_x = value

    def set_visible(self, visible):
        """Set the crosshair visibility.

        Parameters
        ----------
        visible : bool
            True to show the crosshair, False to hide it.
        """
        self.xh.setVisible(visible)
        if hasattr(self, "xh_h"):
            self.xh_h.setVisible(visible)
        if hasattr(self, "xh_v"):
            self.xh_v.setVisible(visible)
